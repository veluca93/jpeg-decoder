#![allow(unsafe_code)]

// This should be its own crate.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// SAFETY for the rest of this impl: a Ssse3 can only be constructed if the CPU supports ssse3,
// making the intrinsic calls safe.

#[inline]
#[target_feature(enable = "ssse3")]
pub fn add_epi8(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_add_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn adds_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_adds_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn subs_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_subs_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn alignr_epi8<const ALIGN: i32>(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_alignr_epi8(a, b, ALIGN) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn mulhrs_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_mulhrs_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn mullo_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_mullo_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn or_si128(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_or_si128(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn packus_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_packus_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn set1_epi16(v: i16) -> __m128i {
    unsafe { _mm_set1_epi16(v) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn set1_epi8(v: i8) -> __m128i {
    unsafe { _mm_set1_epi8(v) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn setr_epi8(
    e15: i8,
    e14: i8,
    e13: i8,
    e12: i8,
    e11: i8,
    e10: i8,
    e9: i8,
    e8: i8,
    e7: i8,
    e6: i8,
    e5: i8,
    e4: i8,
    e3: i8,
    e2: i8,
    e1: i8,
    e0: i8,
) -> __m128i {
    unsafe {
        _mm_setr_epi8(
            e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0,
        )
    }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn setzero_si128() -> __m128i {
    unsafe { _mm_setzero_si128() }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn shuffle_epi8(val: __m128i, shuf: __m128i) -> __m128i {
    unsafe { _mm_shuffle_epi8(val, shuf) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn slli_epi16<const SHIFT: i32>(a: __m128i) -> __m128i {
    unsafe { _mm_slli_epi16(a, SHIFT) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn srai_epi16<const SHIFT: i32>(a: __m128i) -> __m128i {
    unsafe { _mm_srai_epi16(a, SHIFT) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn unpacklo_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_unpacklo_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn unpackhi_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_unpackhi_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn unpacklo_epi32(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_unpacklo_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn unpackhi_epi32(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_unpackhi_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn unpacklo_epi64(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_unpacklo_epi64(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub fn unpackhi_epi64(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_unpackhi_epi64(a, b) }
}

pub fn u8x16tovec(v: [u8; 16]) -> __m128i {
    unsafe { std::mem::transmute(v) }
}

pub fn u16x8tovec(v: [u16; 8]) -> __m128i {
    unsafe { std::mem::transmute(v) }
}

pub fn i16x8tovec(v: [i16; 8]) -> __m128i {
    unsafe { std::mem::transmute(v) }
}

pub fn vectou8x16(v: __m128i) -> [u8; 16] {
    unsafe { std::mem::transmute(v) }
}

pub fn vectou16x8(v: __m128i) -> [u16; 8] {
    unsafe { std::mem::transmute(v) }
}
