use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::fs::File;
use std::io;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;

/// Read-only memory-mapped file region (`mmap(2)` on Linux and macOS) for zero-copy weight loading.
pub struct MmapRegion {
    ptr: NonNull<u8>,
    len: usize,
}

unsafe impl Send for MmapRegion {}
unsafe impl Sync for MmapRegion {}

impl MmapRegion {
    pub fn from_file(file: &File) -> io::Result<Arc<Self>> {
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cannot mmap empty file",
            ));
        }
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            extern "C" {
                fn mmap(
                    addr: *mut std::ffi::c_void,
                    len: usize,
                    prot: i32,
                    flags: i32,
                    fd: i32,
                    offset: i64,
                ) -> *mut std::ffi::c_void;
            }
            const PROT_READ: i32 = 1;
            const MAP_PRIVATE: i32 = 2;
            let raw = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    len,
                    PROT_READ,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if raw == usize::MAX as *mut std::ffi::c_void || raw.is_null() {
                return Err(io::Error::last_os_error());
            }
            let ptr = NonNull::new(raw as *mut u8).unwrap();
            Ok(Arc::new(Self { ptr, len }))
        }
        #[cfg(not(unix))]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "mmap only supported on unix (Linux/macOS)",
            ))
        }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            extern "C" {
                fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
            }
            let _ = munmap(self.ptr.as_ptr() as *mut std::ffi::c_void, self.len);
        }
    }
}

/// 64-byte cache-line aligned buffer for SIMD kernels (ARM64 SME/I8MM/NEON & x86_64 AVX2/VNNI).
/// Supports both owned heap allocations and zero-copy 64-byte-aligned views into an `MmapRegion`.
pub struct AlignedBuf<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
    layout: Option<Layout>,
    _mmap: Option<Arc<MmapRegion>>,
}

unsafe impl<T: Copy + Send> Send for AlignedBuf<T> {}
unsafe impl<T: Copy + Sync> Sync for AlignedBuf<T> {}

impl<T: Copy> AlignedBuf<T> {
    pub const ALIGN: usize = 64;

    #[inline]
    pub fn new_zeroed(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                layout: None,
                _mmap: None,
            };
        }
        let elem_size = std::mem::size_of::<T>().max(1);
        let align = Self::ALIGN.max(std::mem::align_of::<T>());
        let byte_len = (len * elem_size).next_multiple_of(align);
        let layout = Layout::from_size_align(byte_len, align).expect("valid aligned layout");
        let raw = unsafe { alloc_zeroed(layout) as *mut T };
        let ptr = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));
        Self {
            ptr,
            len,
            layout: Some(layout),
            _mmap: None,
        }
    }

    #[inline]
    pub fn from_slice(src: &[T]) -> Self {
        if src.is_empty() {
            return Self::new_zeroed(0);
        }
        let mut buf = Self::new_zeroed(src.len());
        buf.as_mut_slice().copy_from_slice(src);
        buf
    }

    /// Creates a zero-copy 64-byte-aligned read-only view into `mmap` starting at `byte_offset` with `len` elements.
    #[inline]
    pub fn from_mmap(mmap: Arc<MmapRegion>, byte_offset: usize, len: usize) -> io::Result<Self> {
        if len == 0 {
            return Ok(Self::new_zeroed(0));
        }
        let elem_size = std::mem::size_of::<T>().max(1);
        let byte_len = len * elem_size;
        if byte_offset % Self::ALIGN != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "mmap tensor offset is not 64-byte aligned",
            ));
        }
        if byte_offset.saturating_add(byte_len) > mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "mmap tensor extends past end of file",
            ));
        }
        let raw = unsafe { mmap.as_ptr().add(byte_offset) as *mut T };
        let ptr = NonNull::new(raw).unwrap();
        Ok(Self {
            ptr,
            len,
            layout: None,
            _mmap: Some(mmap),
        })
    }

    #[inline]
    pub fn is_mmap(&self) -> bool {
        self._mmap.is_some()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        debug_assert!(
            self._mmap.is_none(),
            "cannot mutably access read-only mmap-backed AlignedBuf"
        );
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        debug_assert!(
            self._mmap.is_none(),
            "cannot mutably access read-only mmap-backed AlignedBuf"
        );
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> Clone for AlignedBuf<T> {
    fn clone(&self) -> Self {
        if let Some(ref mmap) = self._mmap {
            Self {
                ptr: self.ptr,
                len: self.len,
                layout: None,
                _mmap: Some(Arc::clone(mmap)),
            }
        } else {
            Self::from_slice(self.as_slice())
        }
    }
}

impl<T: Copy> Deref for AlignedBuf<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy> DerefMut for AlignedBuf<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}
