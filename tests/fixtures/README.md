# RLE reference excerpts

`rle_reference.json` contains compressed, base64-encoded excerpts from the
original project's `test_seq_rle_high_vol.bin` and `test_seq_rle_low_vol.bin`.
The original-file SHA-256 hashes and excerpt indices are recorded in the JSON.

Each excerpt retains the real 760-byte leading padding, then four consecutive
original frames starting at index 698, followed by a zero delimiter packet.
Pixel-mask hashes were computed independently using the original Python RLE
algorithm, cropping the first 1120 columns and normalizing background to 255.
Tests compare foreground masks, not the vendor's choice of nonzero byte value.

These small fixtures expose the leading-padding regression without depending on
a sibling checkout. They test parsing/reference decoding everywhere and the real
vendor decoder on Windows. They do not simulate the USB/FPGA transport.
