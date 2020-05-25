# ptflops versions log

## v 0.6.2
- Fix integer overflow on Windows.
- Check if the input object is inherited from nn.Module.

## v 0.6.1
- Add experimental version of hooks for recurrent layers (RNN, GRU, LSTM).

## v 0.6
- Add verbose option to log layers that are not supported by ptflops.
- Add an option to filter a list of operations from the final result.

## v 0.5.2
- Fix handling of intermediate dimensions in the Linear layer hook.

## v 0.5
- Add per sequential number of parameters estimation.
- Fix sample doesn't work without GPU.
- Clarified output in sample.

## v 0.4
- Allocate temporal blobs on the same device as model's parameters are located.

## v 0.3
- Add 1d operators: batch norm, poolings, convolution.
- Add ability to output extended report to any output stream.

## v 0.2
- Add new operations: Conv3d, BatchNorm3d, MaxPool3d, AvgPool3d, ConvTranspose2d.
- Add some results on widespread models to the README.
- Minor bugfixes.

## v 0.1
- Initial release with basic functionality
