## Branches

- Code Generator: `f/cmsis-fc`
- uTensor: `cmsis_nn_intg`

## Code Generator command:

`utensor-cli convert deep_mlp1.pb --output-nodes='y_pred' --transform-methods=dropout,quantize,cmsisnn,refcnt,inline`

## Test Inputs Location

`models/test_inputs/`
The file name indicates the hand-written number contained in the file