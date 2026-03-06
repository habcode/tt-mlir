// RUN: ttmlir-opt --tile-matmul -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test: Basic 2D matmul tiling with tile size T=32
// Input: A[64, 128] @ B[128, 96] -> C[64, 96]
// Expected: A_tiled[4, 64, 32] @ B_tiled[4, 32, 96] -> sum([4, 64, 96]) = [64, 96]
func.func @test_basic_matmul_tiling(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
  // CHECK: %0 = "ttir.reshape"(%arg0) <{shape = [4 : i32, 64 : i32, 32 : i32]}> : (tensor<64x128xbf16>) -> tensor<4x64x32xbf16>
  // CHECK: %1 = "ttir.reshape"(%arg1) <{shape = [4 : i32, 32 : i32, 96 : i32]}> : (tensor<128x96xbf16>) -> tensor<4x32x96xbf16>
  // CHECK: %2 = "ttir.matmul"(%0, %1) <{transpose_a = false, transpose_b = false}> : (tensor<4x64x32xbf16>, tensor<4x32x96xbf16>) -> tensor<4x64x96xbf16>
  // CHECK: %3 = "ttir.sum"(%2) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<4x64x96xbf16>) -> tensor<64x96xbf16>
  %result = "ttir.matmul"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<128x96xbf16>) -> tensor<64x96xbf16>
  return %result : tensor<64x96xbf16>
}

// Test: Non-divisible K dimension - should not tile
// K=100 is not divisible by 32
func.func @test_matmul_non_divisible_k(%arg0: tensor<64x100xbf16>, %arg1: tensor<100x96xbf16>) -> tensor<64x96xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<64x100xbf16>, tensor<100x96xbf16>) -> tensor<64x96xbf16>
  %result = "ttir.matmul"(%arg0, %arg1) : (tensor<64x100xbf16>, tensor<100x96xbf16>) -> tensor<64x96xbf16>
  return %result : tensor<64x96xbf16>
}

// Test: 3D matmul (batched) - should not tile  
func.func @test_batched_matmul_no_tile(%arg0: tensor<2x64x128xbf16>, %arg1: tensor<2x128x96xbf16>) -> tensor<2x64x96xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<2x64x128xbf16>, tensor<2x128x96xbf16>) -> tensor<2x64x96xbf16>
  %result = "ttir.matmul"(%arg0, %arg1) : (tensor<2x64x128xbf16>, tensor<2x128x96xbf16>) -> tensor<2x64x96xbf16>
  return %result : tensor<2x64x96xbf16>
}

// Test: Large matmul with K divisible by 32
// Input: A[128, 256] @ B[256, 128] -> C[128, 128]
// K=256, so K/T = 256/32 = 8 tiles
func.func @test_large_matmul(%arg0: tensor<128x256xbf16>, %arg1: tensor<256x128xbf16>) -> tensor<128x128xbf16> {
  // CHECK: %0 = "ttir.reshape"(%arg0) <{shape = [8 : i32, 128 : i32, 32 : i32]}> : (tensor<128x256xbf16>) -> tensor<8x128x32xbf16>
  // CHECK: %1 = "ttir.reshape"(%arg1) <{shape = [8 : i32, 32 : i32, 128 : i32]}> : (tensor<256x128xbf16>) -> tensor<8x32x128xbf16>
  // CHECK: %2 = "ttir.matmul"(%0, %1) <{transpose_a = false, transpose_b = false}> : (tensor<8x128x32xbf16>, tensor<8x32x128xbf16>) -> tensor<8x128x128xbf16>
  // CHECK: %3 = "ttir.sum"(%2) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<8x128x128xbf16>) -> tensor<128x128xbf16>
  %result = "ttir.matmul"(%arg0, %arg1) : (tensor<128x256xbf16>, tensor<256x128xbf16>) -> tensor<128x128xbf16>
  return %result : tensor<128x128xbf16>
}

// Test: Small matmul with exact tile match
// Input: A[32, 32] @ B[32, 64] -> C[32, 64]
// K=32, so K/T = 32/32 = 1 tile
func.func @test_small_matmul_exact_tile(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x64xbf16>) -> tensor<32x64xbf16> {
  // CHECK: %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 32 : i32, 32 : i32]}> : (tensor<32x32xbf16>) -> tensor<1x32x32xbf16>
  // CHECK: %1 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 32 : i32, 64 : i32]}> : (tensor<32x64xbf16>) -> tensor<1x32x64xbf16>
  // CHECK: %2 = "ttir.matmul"(%0, %1) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x32xbf16>, tensor<1x32x64xbf16>) -> tensor<1x32x64xbf16>
  // CHECK: %3 = "ttir.sum"(%2) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<1x32x64xbf16>) -> tensor<32x64xbf16>
  %result = "ttir.matmul"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
  return %result : tensor<32x64xbf16>
}
