#![no_std]

//! # AZTEC
//!
//! A compression algorithm based on: <https://ieeexplore.ieee.org/document/4502549>
//!
//! 3 bytes get used for each line/slope which means that in the worst case the compressed data
//! will be 1.5 times larger than the original.
//!
//!
//! Usage:
//! ```rust
//! const THRESHOLD: u16 = 50;
//! let config = AztecConfigBuilder::new(THRESHOLD)
//!    .max_slope_line_len(4)
//!    .build()
//!    .expect("Invalid Config");
//! ```
//!

struct Buffer<'a> {
    data: &'a [u8],
    current: usize,
}

impl<'a> Buffer<'a> {
    fn new(data: &'a [u8]) -> Buffer<'a> {
        Buffer { data, current: 0 }
    }
}

struct BufferMut<'a> {
    data: &'a mut [u8],
    current: usize,
}

impl<'a> BufferMut<'a> {
    fn new(data: &'a mut [u8]) -> BufferMut<'a> {
        BufferMut { data, current: 0 }
    }
}

struct Packet {
    slope: bool,
    length: u8,
    data: u16,
}

#[derive(Debug)]
pub enum AztecError {
    MaxLineLenTooLong,
    MaxSlopeLenTooLong,
    MaxSlopeLineLenTooLong,
    ProvidedBufferIsNotLongEnough,
}

const NOT_LONG: AztecError = AztecError::ProvidedBufferIsNotLongEnough;

impl Packet {
    /// Create a line packet
    fn line(length: u8, data: u16) -> Self {
        // println!("Created Line. LEN: {} DATA: {}", length, data);
        Self {
            slope: false,
            length,
            data,
        }
    }

    /// Create a slope packet
    fn slope(length: u8, data: u16) -> Self {
        // println!("Created Slope. LEN: {} DATA: {}", length, data);
        Self {
            slope: true,
            length,
            data,
        }
    }

    /// Write the packet to a buffer
    fn write(&self, buf: &mut BufferMut) -> Result<(), AztecError> {
        let i = buf.current;

        let val = ((self.slope as u8) << 7) | self.length;
        let [byte1, byte2] = self.data.to_le_bytes();

        *buf.data.get_mut(i).ok_or(NOT_LONG)? = val;
        *buf.data.get_mut(i + 1).ok_or(NOT_LONG)? = byte1;
        *buf.data.get_mut(i + 2).ok_or(NOT_LONG)? = byte2;

        buf.current += 3;
        Ok(())
    }

    /// Parse the packet from bytes
    fn from_buffer(buf: &mut Buffer) -> Result<Self, AztecError> {
        let i = buf.current;

        let x = buf.data.get(i).ok_or(NOT_LONG)?;
        let byte1 = buf.data.get(i + 1).ok_or(NOT_LONG)?;
        let byte2 = buf.data.get(i + 2).ok_or(NOT_LONG)?;

        let slope = (x >> 7) == 1;
        let length = x & 0b111_1111;
        let data = u16::from_le_bytes([*byte1, *byte2]);

        buf.current += 3;

        Ok(Self {
            slope,
            length,
            data,
        })
    }
}

/// Configuration for the AZTEC compression algorithm. 
/// Use [`AztecConfigBuilder`] in order to create it.
pub struct AztecConfig {
    threshold: u16,
    max_line_len: u8,
    max_slope_len: u8,
    max_slope_line_len: u8,
}

impl AztecConfig {
    /// Get the current threshold
    pub fn threshold(&self) -> u16 {
        self.threshold
    }

    /// Get the current max_line_len
    pub fn max_line_len(&self) -> u8 {
        self.max_line_len
    }

    /// Get the current max_slope_len
    pub fn max_slope_len(&self) -> u8 {
        self.max_slope_len
    }

    /// Get the current max_slope_line_len
    pub fn max_slope_line_len(&self) -> u8 {
        self.max_slope_line_len
    }
}

/// Builder for the AZTEC configuration
pub struct AztecConfigBuilder {
    config: AztecConfig,
}

impl AztecConfigBuilder {
    /// Create a new builder with a given threshold
    pub fn new(threshold: u16) -> Self {
        Self {
            config: AztecConfig {
                threshold,
                max_line_len: 63,
                max_slope_len: 63,
                max_slope_line_len: 4,
            },
        }
    }

    /// The threshold for a series of samples to be represented as a line
    pub fn threshold(mut self, threshold: u16) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// The maximum length of a line (in samples).
    ///
    /// The maximum allowed value is u7 (63)
    pub fn max_line_len(mut self, max_line_len: u8) -> Self {
        self.config.max_line_len = max_line_len;
        self
    }

    /// The maximum length of a slope (in samples). 
    /// Set this value lower if the slopes are too long.
    ///
    /// The maximum allowed value is u7 (63)
    pub fn max_slope_len(mut self, max_slope_len: u8) -> Self {
        self.config.max_slope_len = max_slope_len;
        self
    }

    /// The maximum length of a line for it to be considered a part of the slope.
    ///
    /// The value should be less than max_slope_len
    pub fn max_slope_line_len(mut self, max_slope_line_len: u8) -> Self {
        self.config.max_slope_line_len = max_slope_line_len;
        self
    }

    /// Validates the given parameters and creates a new AztecConfig
    pub fn build(self) -> Result<AztecConfig, AztecError> {
        if self.config.max_line_len > 63 {
            return Err(AztecError::MaxLineLenTooLong);
        }

        if self.config.max_slope_len > 63 {
            return Err(AztecError::MaxSlopeLenTooLong);
        }

        if self.config.max_slope_line_len >= self.config.max_slope_len {
            return Err(AztecError::MaxSlopeLineLenTooLong);
        }

        Ok(self.config)
    }
}

enum SlopeState {
    NoSlope,
    OngoingSlope(Option<bool>),
    SlopeDone,
}

/// Compress the given data
///
/// The result buffer should be at least 3 times the length (1.5 times the size in memory) of the
/// source buffer to account for the worst case scenario where every sample is a line/slope
pub fn compress(
    data: &[u16],
    result: &mut [u8],
    config: &AztecConfig,
) -> Result<usize, AztecError> {
    // Will only reach the capacity if every single sample is a line
    let mut buf = BufferMut::new(result);

    let mut x = data[0];
    let mut min = x;
    let mut max = x;
    let mut old_min = x;
    let mut old_max = x;
    Packet::line(1, x).write(&mut buf)?;

    let mut slope_state = SlopeState::NoSlope;
    let mut slope_length = 0;

    let mut line_done = false;
    let mut line_length = 1;
    let mut line_value;
    let mut previous_line_value = 0;

    let mut i = 1;

    while i < data.len() {
        let mut increment = true;

        x = data[i];

        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }

        // println!("i-{} x-{} MIN: {} MAX: {}", i, x, old_min, old_max);
        line_value = ((old_max as u32 + old_min as u32) / 2) as u16;

        // Terminate the line if it is too long or the limits exceed the threshold
        if line_length >= config.max_line_len || max - min > config.threshold {
            line_done = true;
        }

        // Terminate the line and the slope on the last sample
        let last_sample = i == data.len() - 1;
        if last_sample {
            line_done = true;
            if let SlopeState::OngoingSlope(_) = slope_state {
                slope_state = SlopeState::SlopeDone;
            }
        }

        if line_done {
            // println!("i-{} LINE LEN: {} VAL: {}", i, line_length, line_value);

            // Check if the line could be a slope
            // Do not change the slope state on the last sample
            if !last_sample {
                if line_length <= config.max_slope_line_len {
                    match slope_state {
                        SlopeState::NoSlope => {
                            // Start a new slope
                            slope_state = SlopeState::OngoingSlope(None);
                            // println!("i-{} SLOPE START", i);
                        }
                        SlopeState::OngoingSlope(slope_dir) => {
                            let diff = line_value as i32 - previous_line_value as i32;
                            let mut sign_changed = false;
                            // println!("DIFF: {}", diff);
                            let current_slope_dir = if diff == 0 { None } else { Some(diff > 0) };
                            if let Some(current_slope_dir) = current_slope_dir {
                                if let Some(slope_dir) = slope_dir {
                                    // Check if the slope direction still matches
                                    if current_slope_dir != slope_dir {
                                        sign_changed = true;
                                    }
                                }

                                if sign_changed {
                                    slope_state = SlopeState::SlopeDone;
                                } else {
                                    slope_state = SlopeState::OngoingSlope(Some(current_slope_dir));
                                }
                            }
                        }
                        SlopeState::SlopeDone => {}
                    }
                } else {
                    // Terminate a slope if there is a line longer than the maximum allowed
                    if let SlopeState::OngoingSlope(_) = slope_state {
                        slope_state = SlopeState::SlopeDone;
                    }
                }
            }

            // Terminate a slope that will be too long
            if let SlopeState::OngoingSlope(_) = slope_state {
                if slope_length as u16 + line_length as u16 >= config.max_slope_len as u16 {
                    slope_state = SlopeState::SlopeDone;
                }
            }

            // println!("OUT {} {:?}", line_length, slope_state);
            match slope_state {
                // Only write out the line
                SlopeState::NoSlope => {
                    Packet::line(line_length, line_value).write(&mut buf)?;
                }
                // This line is a part of the slope
                SlopeState::OngoingSlope(_) => {
                    slope_length += line_length;
                }
                SlopeState::SlopeDone => {
                    // println!("SLOPE END {}", i);
                    Packet::slope(slope_length, previous_line_value).write(&mut buf)?;

                    // Process the current line again because it could potentially be a part of a new slope
                    increment = false;
                    slope_state = SlopeState::NoSlope;
                    slope_length = 0;
                }
            }

            // Reset to initial state after a line
            if increment {
                previous_line_value = line_value;
                min = x;
                max = x;
                line_done = false;
                line_length = 1;
            }
        } else {
            line_length += 1;
        }

        if increment {
            old_min = min;
            old_max = max;
            i += 1;
        }
    }

    Ok(buf.current + 1)
}

/// Decompress the given data
pub fn decompress(data: &[u8], output: &mut [u16]) -> Result<usize, AztecError> {
    let mut buf = Buffer::new(data);

    let mut output_i = 0;
    let mut previous_line_value: u16 = 0;

    // Each packet is 3 bytes
    for _ in 0..data.len() / 3 {
        let packet = Packet::from_buffer(&mut buf)?;
        if packet.slope {
            if packet.length == 1 {
                output[output_i] = packet.data;
                output_i += 1;
            } else {
                for i in 0..packet.length {
                    // Linear interpolation
                    let value = previous_line_value as i32
                        + ((i as i32) * (packet.data as i32 - previous_line_value as i32))
                            / (packet.length as i32 - 1);
                    output[output_i] = value as u16;
                    output_i += 1;
                }
            }
        } else {
            for _ in 0..packet.length {
                output[output_i] = packet.data;
                output_i += 1;
            }
        }

        previous_line_value = packet.data;
    }

    Ok(output_i)
}

#[cfg(test)]
mod tests {
    use crate::*;

    const LEN: usize = 128733;
    const RES_LEN: usize = LEN * 3;

    #[test]
    // A flat line will be compressed losslessly
    fn straight_line() {
        let data = [128; LEN];
        let mut decompressed = [0; LEN];

        let mut result = [0; RES_LEN];
        let config = AztecConfigBuilder::new(10)
            .max_slope_line_len(4)
            .build()
            .unwrap();
        let len = compress(&data, &mut result, &config).unwrap();
        let (compressed, _) = result.split_at(len);

        let len = decompress(&compressed, &mut decompressed).unwrap();

        assert_eq!(len, LEN);
        // A straight line will be compressed without any loss
        assert_eq!(data, decompressed);
    }

    #[test]
    fn straight_line_long_slope() {
        let data = [128; LEN];
        let mut decompressed = [0; LEN];

        let mut result = [0; RES_LEN];
        let config = AztecConfigBuilder::new(10)
            .max_slope_line_len(62)
            .build()
            .unwrap();
        let len = compress(&data, &mut result, &config).unwrap();
        let (compressed, _) = result.split_at(len);

        let len = decompress(&compressed, &mut decompressed).unwrap();

        assert_eq!(len, LEN);
        assert_eq!(data, decompressed);
    }

    #[test]
    fn triangle() {
        let mut data = [0; LEN];

        let mut val: u16 = 0;
        for i in 0..LEN {
            data[i] = val;

            if val == u16::MAX {
                val = 0;
            } else {
                val += 1;
            }
        }

        let mut decompressed = [0; LEN * 3];

        let mut result = [0; RES_LEN];
        let config = AztecConfigBuilder::new(10)
            .max_slope_line_len(4)
            .build()
            .unwrap();
        let len = compress(&data, &mut result, &config).unwrap();
        let (compressed, _) = result.split_at(len);

        let len = decompress(&compressed, &mut decompressed).unwrap();

        assert_eq!(len, LEN);
    }

    #[test]
    fn multiple_random() {
        const RANDOM_TESTS: usize = 20;

        let mut data = [0; LEN];

        for _ in 0..RANDOM_TESTS {
            for x in &mut data {
                *x = rand::random();
            }

            let mut decompressed = [0; LEN * 3];

            let mut result = [0; RES_LEN];
            let config = AztecConfigBuilder::new(10)
                .max_slope_line_len(4)
                .build()
                .unwrap();
            let len = compress(&data, &mut result, &config).unwrap();
            let (compressed, _) = result.split_at(len);

            let len = decompress(&compressed, &mut decompressed).unwrap();

            assert_eq!(len, LEN);
        }
    }
}
