#![allow(unsafe_code)]

mod neon;
mod ssse3;
mod ssse3_wrapper;

/// Arch-specific implementation of YCbCr conversion. Returns the number of pixels that were
/// converted.
pub fn get_color_convert_line_ycbcr() -> Option<fn(&[u8], &[u8], &[u8], &mut [u8]) -> usize> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[allow(unsafe_code)]
    {
        if is_x86_feature_detected!("ssse3") {
            fn fun(y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8]) -> usize {
                unsafe {
                    // SAFETY: ssse3 presence is already checked by if condition.
                    ssse3::color_convert_line_ycbcr(y, cb, cr, out)
                }
            }
            return Some(fun);
        }
    }
    // Runtime detection is not needed on aarch64.
    #[cfg(all(feature = "nightly_aarch64_neon", target_arch = "aarch64"))]
    {
        return Some(neon::color_convert_line_ycbcr);
    }
    #[allow(unreachable_code)]
    None
}

/// Arch-specific implementation of 8x8 IDCT.
pub fn get_dequantize_and_idct_block_8x8() -> Option<fn(&[i16; 64], &[u16; 64], usize, &mut [u8])> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[allow(unsafe_code)]
    {
        if is_x86_feature_detected!("ssse3") {
            fn fun(
                coefficients: &[i16; 64],
                quantization_table: &[u16; 64],
                output_linestride: usize,
                output: &mut [u8],
            ) {
                unsafe {
                    // SAFETY: ssse3 presence is already checked by if condition.
                    ssse3::dequantize_and_idct_block_8x8(
                        coefficients,
                        quantization_table,
                        output_linestride,
                        output,
                    )
                }
            }
            return Some(fun);
        }
    }
    // Runtime detection is not needed on aarch64.
    #[cfg(all(feature = "nightly_aarch64_neon", target_arch = "aarch64"))]
    {
        return Some(neon::dequantize_and_idct_block_8x8);
    }
    #[allow(unreachable_code)]
    None
}
