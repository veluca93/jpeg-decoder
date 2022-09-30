use crate::arch::ssse3_wrapper::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::__m128i;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__m128i;

#[target_feature(enable = "ssse3")]
fn idct8(data: &mut [__m128i; 8]) {
    // The fixed-point constants here are obtained by taking the fractional part of the constants
    // from the non-SIMD implementation and scaling them up by 1<<15. This is because
    // mulhrs_epi16(a, b) is effectively equivalent to (a*b)>>15 (except for possibly some
    // slight differences in rounding).

    // The code here is effectively equivalent to the calls to "kernel" in idct.rs, except that it
    // doesn't apply any further scaling and fixed point constants have a different precision.

    let p2 = data[2];
    let p3 = data[6];
    let p1 = mulhrs_epi16(adds_epi16(p2, p3), set1_epi16(17734)); // 0.5411961
    let t2 = subs_epi16(
        subs_epi16(p1, p3),
        mulhrs_epi16(p3, set1_epi16(27779)), // 0.847759065
    );
    let t3 = adds_epi16(p1, mulhrs_epi16(p2, set1_epi16(25079))); // 0.765366865

    let p2 = data[0];
    let p3 = data[4];
    let t0 = adds_epi16(p2, p3);
    let t1 = subs_epi16(p2, p3);

    let x0 = adds_epi16(t0, t3);
    let x3 = subs_epi16(t0, t3);
    let x1 = adds_epi16(t1, t2);
    let x2 = subs_epi16(t1, t2);

    let t0 = data[7];
    let t1 = data[5];
    let t2 = data[3];
    let t3 = data[1];

    let p3 = adds_epi16(t0, t2);
    let p4 = adds_epi16(t1, t3);
    let p1 = adds_epi16(t0, t3);
    let p2 = adds_epi16(t1, t2);
    let p5 = adds_epi16(p3, p4);
    let p5 = adds_epi16(p5, mulhrs_epi16(p5, set1_epi16(5763))); // 0.175875602

    let t0 = mulhrs_epi16(t0, set1_epi16(9786)); // 0.298631336
    let t1 = adds_epi16(
        adds_epi16(t1, t1),
        mulhrs_epi16(t1, set1_epi16(1741)), // 0.053119869
    );
    let t2 = adds_epi16(
        adds_epi16(t2, adds_epi16(t2, t2)),
        mulhrs_epi16(t2, set1_epi16(2383)), // 0.072711026
    );
    let t3 = adds_epi16(t3, mulhrs_epi16(t3, set1_epi16(16427))); // 0.501321110

    let p1 = subs_epi16(p5, mulhrs_epi16(p1, set1_epi16(29490))); // 0.899976223
    let p2 = subs_epi16(
        subs_epi16(subs_epi16(p5, p2), p2),
        mulhrs_epi16(p2, set1_epi16(18446)), // 0.562915447
    );

    let p3 = subs_epi16(
        mulhrs_epi16(p3, set1_epi16(-31509)), // -0.961570560
        p3,
    );
    let p4 = mulhrs_epi16(p4, set1_epi16(-12785)); // -0.390180644

    let t3 = adds_epi16(adds_epi16(p1, p4), t3);
    let t2 = adds_epi16(adds_epi16(p2, p3), t2);
    let t1 = adds_epi16(adds_epi16(p2, p4), t1);
    let t0 = adds_epi16(adds_epi16(p1, p3), t0);

    data[0] = adds_epi16(x0, t3);
    data[7] = subs_epi16(x0, t3);
    data[1] = adds_epi16(x1, t2);
    data[6] = subs_epi16(x1, t2);
    data[2] = adds_epi16(x2, t1);
    data[5] = subs_epi16(x2, t1);
    data[3] = adds_epi16(x3, t0);
    data[4] = subs_epi16(x3, t0);
}

#[target_feature(enable = "ssse3")]
fn transpose8(data: &mut [__m128i; 8]) {
    // Transpose a 8x8 matrix with a sequence of interleaving operations.
    // Naming: dABl contains elements from the *l*ower halves of vectors A and B, interleaved, i.e.
    // A0 B0 A1 B1 ...
    // dABCDll contains elements from the lower quarter (ll) of vectors A, B, C, D, interleaved -
    // A0 B0 C0 D0 A1 B1 C1 D1 ...
    let d01l = unpacklo_epi16(data[0], data[1]);
    let d23l = unpacklo_epi16(data[2], data[3]);
    let d45l = unpacklo_epi16(data[4], data[5]);
    let d67l = unpacklo_epi16(data[6], data[7]);
    let d01h = unpackhi_epi16(data[0], data[1]);
    let d23h = unpackhi_epi16(data[2], data[3]);
    let d45h = unpackhi_epi16(data[4], data[5]);
    let d67h = unpackhi_epi16(data[6], data[7]);
    // Operating on 32-bits will interleave *consecutive pairs* of 16-bit integers.
    let d0123ll = unpacklo_epi32(d01l, d23l);
    let d0123lh = unpackhi_epi32(d01l, d23l);
    let d4567ll = unpacklo_epi32(d45l, d67l);
    let d4567lh = unpackhi_epi32(d45l, d67l);
    let d0123hl = unpacklo_epi32(d01h, d23h);
    let d0123hh = unpackhi_epi32(d01h, d23h);
    let d4567hl = unpacklo_epi32(d45h, d67h);
    let d4567hh = unpackhi_epi32(d45h, d67h);
    // Operating on 64-bits will interleave *consecutive quadruples* of 16-bit integers.
    data[0] = unpacklo_epi64(d0123ll, d4567ll);
    data[1] = unpackhi_epi64(d0123ll, d4567ll);
    data[2] = unpacklo_epi64(d0123lh, d4567lh);
    data[3] = unpackhi_epi64(d0123lh, d4567lh);
    data[4] = unpacklo_epi64(d0123hl, d4567hl);
    data[5] = unpackhi_epi64(d0123hl, d4567hl);
    data[6] = unpacklo_epi64(d0123hh, d4567hh);
    data[7] = unpackhi_epi64(d0123hh, d4567hh);
}

#[inline(always)]
fn store8u8low(vec: __m128i, data: &mut [u8]) {
    let vec: [u8; 16] = vectou8x16(vec);
    data[..8].copy_from_slice(&vec[..8]);
}

#[inline]
#[target_feature(enable = "ssse3")]
fn load64i16(data_in: &[i16; 64]) -> [__m128i; 8] {
    let mut data: [[i16; 8]; 8] = [[0; 8]; 8];
    for i in 0..8 {
        data[i].copy_from_slice(&data_in[i * 8..i * 8 + 8]);
    }
    let mut out: [__m128i; 8] = [setzero_si128(); 8];
    for i in 0..8 {
        out[i] = i16x8tovec(data[i]);
    }
    out
}

#[inline]
#[target_feature(enable = "ssse3")]
fn load64u16(data_in: &[u16; 64]) -> [__m128i; 8] {
    let mut data: [[u16; 8]; 8] = [[0; 8]; 8];
    for i in 0..8 {
        data[i].copy_from_slice(&data_in[i * 8..i * 8 + 8]);
    }
    let mut out: [__m128i; 8] = [setzero_si128(); 8];
    for i in 0..8 {
        out[i] = u16x8tovec(data[i]);
    }
    out
}

#[target_feature(enable = "ssse3")]
pub fn dequantize_and_idct_block_8x8(
    coefficients: &[i16; 64],
    quantization_table: &[u16; 64],
    output_linestride: usize,
    output: &mut [u8],
) {
    let coefficients = load64i16(coefficients);
    let quantization_table = load64u16(quantization_table);

    const SHIFT: i32 = 3;

    // Dequantize and scale DCT coefficients.
    let mut data = [setzero_si128(); 8];
    for i in 0..8 {
        data[i] = slli_epi16::<SHIFT>(mullo_epi16(coefficients[i], quantization_table[i]));
    }

    // Usual column IDCT - transpose - column IDCT - transpose approach.
    idct8(&mut data);
    transpose8(&mut data);
    idct8(&mut data);
    transpose8(&mut data);

    for i in 0..8 {
        // The two passes of the IDCT algorithm give us a factor of 8, so the shift here is
        // increased by 3.
        // As values will be stored in a u8, they need to be 128-centered and not 0-centered.
        // We add 128 with the appropriate shift for that purpose.
        const OFFSET: i16 = 128 << (SHIFT + 3);
        // We want rounding right shift, so we should add (1/2) << (SHIFT+3) before shifting.
        const ROUNDING_BIAS: i16 = (1 << (SHIFT + 3)) >> 1;

        let data_with_offset = adds_epi16(data[i], set1_epi16(OFFSET + ROUNDING_BIAS));

        const SHIFT3: i32 = SHIFT + 3;

        let val = packus_epi16(srai_epi16::<SHIFT3>(data_with_offset), setzero_si128());
        store8u8low(val, &mut output[output_linestride * i..]);
    }
}

#[inline]
fn load8u8low(data_in: &[u8]) -> __m128i {
    let mut data: [u8; 16] = [0; 16];
    data[0..8].copy_from_slice(&data_in[0..8]);
    u8x16tovec(data)
}

#[inline]
fn store16u8(vec: __m128i, data: &mut [u8]) {
    let vec: [u8; 16] = vectou8x16(vec);
    data[..16].copy_from_slice(&vec[..16]);
}

#[target_feature(enable = "ssse3")]
pub fn color_convert_line_ycbcr(y: &[u8], cb: &[u8], cr: &[u8], output: &mut [u8]) -> usize {
    let mut num_processed = 0;

    for ((y, cb), (cr, output)) in y
        .chunks_exact(8)
        .zip(cb.chunks_exact(8))
        .zip(cr.chunks_exact(8).zip(output.chunks_exact_mut(24)))
    {
        let y = load8u8low(&y);
        let cb = load8u8low(&cb);
        let cr = load8u8low(&cr);

        const SHIFT: i32 = 6;
        // Convert to 16 bit.
        let shuf16 = setr_epi8(
            0, -0x7F, 1, -0x7F, 2, -0x7F, 3, -0x7F, 4, -0x7F, 5, -0x7F, 6, -0x7F, 7, -0x7F,
        );
        let y = slli_epi16::<SHIFT>(shuffle_epi8(y, shuf16));
        let cb = slli_epi16::<SHIFT>(shuffle_epi8(cb, shuf16));
        let cr = slli_epi16::<SHIFT>(shuffle_epi8(cr, shuf16));

        // Add offsets
        let c128 = set1_epi16(128 << SHIFT);
        let y = adds_epi16(y, set1_epi16((1 << SHIFT) >> 1));
        let cb = subs_epi16(cb, c128);
        let cr = subs_epi16(cr, c128);

        // Compute cr * 1.402, cb * 0.34414, cr * 0.71414, cb * 1.772
        let cr_140200 = adds_epi16(mulhrs_epi16(cr, set1_epi16(13173)), cr);
        let cb_034414 = mulhrs_epi16(cb, set1_epi16(11276));
        let cr_071414 = mulhrs_epi16(cr, set1_epi16(23401));
        let cb_177200 = adds_epi16(mulhrs_epi16(cb, set1_epi16(25297)), cb);

        // Last conversion step.
        let r = adds_epi16(y, cr_140200);
        let g = subs_epi16(y, adds_epi16(cb_034414, cr_071414));
        let b = adds_epi16(y, cb_177200);

        // Shift back and convert to u8.
        let zero = setzero_si128();
        let r = packus_epi16(srai_epi16::<SHIFT>(r), zero);
        let g = packus_epi16(srai_epi16::<SHIFT>(g), zero);
        let b = packus_epi16(srai_epi16::<SHIFT>(b), zero);

        // Shuffle rrrrrrrrggggggggbbbbbbbb to rgbrgbrgb...

        // Control vectors for shuffle_epi8. -0x7F is selected so that the resulting position
        // after shuffle_epi8 will be filled with 0, so that the r, g, and b vectors can then
        // be OR-ed together.
        let shufr = setr_epi8(
            0, -0x7F, -0x7F, 1, -0x7F, -0x7F, 2, -0x7F, -0x7F, 3, -0x7F, -0x7F, 4, -0x7F, -0x7F, 5,
        );
        let shufg = setr_epi8(
            -0x7F, 0, -0x7F, -0x7F, 1, -0x7F, -0x7F, 2, -0x7F, -0x7F, 3, -0x7F, -0x7F, 4, -0x7F,
            -0x7F,
        );
        let shufb = alignr_epi8::<15>(shufg, shufg);

        let rgb_low = or_si128(
            shuffle_epi8(r, shufr),
            or_si128(shuffle_epi8(g, shufg), shuffle_epi8(b, shufb)),
        );

        // For the next part of the rgb vectors, we need to select R values from 6 up, G and B from
        // 5 up. The highest bit of -0x7F + 6 is still set, so the corresponding location will
        // still be 0.
        let shufr1 = add_epi8(shufb, set1_epi8(6));
        let shufg1 = add_epi8(shufr, set1_epi8(5));
        let shufb1 = add_epi8(shufg, set1_epi8(5));

        let rgb_hi = or_si128(
            shuffle_epi8(r, shufr1),
            or_si128(shuffle_epi8(g, shufg1), shuffle_epi8(b, shufb1)),
        );
        store16u8(rgb_low, &mut output[..]);
        store8u8low(rgb_hi, &mut output[16..]);
        num_processed += 8;
    }

    num_processed
}
