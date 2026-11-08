use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;

/// 64-byte cache-line aligned, owned buffer for SIMD kernels (ARM64 SME/I8MM/NEON & x86_64 AVX2/VNNI).
pub struct AlignedBuf<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
    layout: Option<Layout>,
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
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> Clone for AlignedBuf<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
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
