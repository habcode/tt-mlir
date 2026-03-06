// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Transforms/Passes.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::transforms {

#define GEN_PASS_DEF_TILEMATMUL
#include "ttmlir/Transforms/Passes.h.inc"

namespace {

/// Pattern to rewrite a 2D ttir.matmul into a tiled sequence:
///  - reshape inputs into tiles
///  - perform a batched matmul over tiles
///  - reduce (sum) across the tile dimension to produce the final 2D result
///
/// Example: Tile size T
///   A[M,K] -> A_tiled[K/T, M, T]
///   B[K,N] -> B_tiled[K/T, T, N]
///   batched_matmul: [K/T, M, T] @ [K/T, T, N] -> [K/T, M, N]
///   sum over dim0 -> [M, N]
class MatmulTilingPattern : public OpRewritePattern<ttir::MatmulOp> {
private:
  static constexpr int64_t TILE_SIZE = 32; // TODO(habcode): make this configurable

  bool is2DMatmul(ttir::MatmulOp matmulOp) const {
    auto a_type = mlir::dyn_cast<RankedTensorType>(matmulOp.getA().getType());
    auto b_type = mlir::dyn_cast<RankedTensorType>(matmulOp.getB().getType());

    if (!a_type || !b_type) {
      return false;
    }

    return a_type.getRank() == 2 && b_type.getRank() == 2;
  }

  // Create a reshape op that tiles the matmul operand A [M, K] into [K/T, M, T].
  ttir::ReshapeOp createATiledReshape(PatternRewriter &rewriter,
                                        ttir::MatmulOp matmulOp) const {
    Location loc = matmulOp.getLoc();
    auto a_type = mlir::dyn_cast<RankedTensorType>(matmulOp.getA().getType());
    auto elementType = a_type.getElementType();

    int64_t m = a_type.getShape()[0];
    int64_t k = a_type.getShape()[1];

    // Reshape [M, K] -> [K/T, M, T]
    int64_t numTiles = k / TILE_SIZE;
    SmallVector<int64_t> tiledShape = {numTiles, m, TILE_SIZE};

    // Build a RankedTensorType for the tiled A preserving any encoding.
    auto tiledAType =
        RankedTensorType::get(tiledShape, elementType, a_type.getEncoding());
    
    // Create an I32ArrayAttr describing the reshape target shape for the op.
    auto shapeAttr = rewriter.getI32ArrayAttr(
        SmallVector<int32_t>{static_cast<int32_t>(numTiles),
                             static_cast<int32_t>(m),
                             static_cast<int32_t>(TILE_SIZE)});

    auto a_tiled_reshape = rewriter.create<ttir::ReshapeOp>(
        loc, tiledAType, matmulOp.getA(), shapeAttr);

    return a_tiled_reshape;
  }

  // Create a reshape op that tiles the matmul operand B [K, N] into [K/T, T, N].
  ttir::ReshapeOp createBTiledReshape(PatternRewriter &rewriter,
                                        ttir::MatmulOp matmulOp) const {
    Location loc = matmulOp.getLoc();
    auto b_type = mlir::dyn_cast<RankedTensorType>(matmulOp.getB().getType());
    auto elementType = b_type.getElementType();

    int64_t k = b_type.getShape()[0];
    int64_t n = b_type.getShape()[1];

    // Reshape [K, N] -> [K/T, T, N]
    int64_t numTiles = k / TILE_SIZE;
    SmallVector<int64_t> tiledShape = {numTiles, TILE_SIZE, n};

    // Build a RankedTensorType for the tiled B preserving any encoding.
    auto tiledBType =
        RankedTensorType::get(tiledShape, elementType, b_type.getEncoding());
    
    // Create the shape attribute for the B reshape op.
    auto shapeAttr = rewriter.getI32ArrayAttr(
        SmallVector<int32_t>{static_cast<int32_t>(numTiles),
                             static_cast<int32_t>(TILE_SIZE),
                             static_cast<int32_t>(n)});

    auto b_tiled_reshape = rewriter.create<ttir::ReshapeOp>(
        loc, tiledBType, matmulOp.getB(), shapeAttr);

    return b_tiled_reshape;
  }

  // Create a batched matmul that consumes the tiled operands and produces
  // a tensor shaped [K/T, M, N].
  ttir::MatmulOp createTiledMatmul(PatternRewriter &rewriter,
                                   ttir::MatmulOp matmulOp,
                                   ttir::ReshapeOp a_tiled_reshape,
                                   ttir::ReshapeOp b_tiled_reshape) const {
    Location loc = matmulOp.getLoc();
    auto a_type = mlir::dyn_cast<RankedTensorType>(a_tiled_reshape.getResult().getType());
    auto b_type = mlir::dyn_cast<RankedTensorType>(b_tiled_reshape.getResult().getType());
    auto elementType = a_type.getElementType();

    // Result shape: [K/T, M, N]
    SmallVector<int64_t> matmulResultShape = {
        a_type.getShape()[0],  // K/T (batch dimension)
        a_type.getShape()[1],  // M
        b_type.getShape()[2]   // N
    };

    // Create a RankedTensorType for the matmul result.
    auto matmulResultType = RankedTensorType::get(matmulResultShape, elementType);

    // Create the tiled matmul. Forward transpose attributes from the original
    // matmul to preserve semantics (no-op if both are false).
    auto tiledMatmul = rewriter.create<ttir::MatmulOp>(
        loc, matmulResultType, a_tiled_reshape.getResult(), b_tiled_reshape.getResult(),
        matmulOp.getTransposeA(), matmulOp.getTransposeB());

    return tiledMatmul;
  }

  /// Create sum/reduce to collapse tile dimension
  /// [K/T, M, N] -> [M, N] by summing over dimension 0
  ttir::SumOp createTiledReduce(PatternRewriter &rewriter,
                                ttir::MatmulOp matmulOp,
                                ttir::MatmulOp tiledMatmul) const {
    Location loc = matmulOp.getLoc();
    auto resultType = mlir::dyn_cast<RankedTensorType>(matmulOp.getResult().getType());
    auto elementType = resultType.getElementType();

    auto tiledMatmulType =
        mlir::dyn_cast<RankedTensorType>(tiledMatmul.getResult().getType());
    
    // Result shape: [M, N]
    SmallVector<int64_t> finalShape = {
        tiledMatmulType.getShape()[1],  // M
        tiledMatmulType.getShape()[2]   // N
    };

    auto finalType = RankedTensorType::get(finalShape, elementType);
    
    // Create dim_arg as I32ArrayAttr with dim=0 (sum over K/T dimension)
    auto dimArgAttr = rewriter.getI32ArrayAttr(SmallVector<int32_t>{0});
    
    // Create the SumOp: arguments are (loc, resultType, input, keep_dim, dim_arg).
    // keep_dim=false means the reduced dimension is removed from the shape.
    auto finalSum = rewriter.create<ttir::SumOp>(
        loc, finalType, tiledMatmul.getResult(),
        rewriter.getBoolAttr(false), dimArgAttr);

    return finalSum;
  }

public:
  using OpRewritePattern<ttir::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    // Only tile 2D matmuls
    if (!is2DMatmul(matmulOp)) {
      return rewriter.notifyMatchFailure(matmulOp,
                                         "matmul is not 2D");
    }

    // Check if K dimension is divisible by TILE_SIZE
    auto a_type = mlir::dyn_cast<RankedTensorType>(matmulOp.getA().getType());
    auto b_type = mlir::dyn_cast<RankedTensorType>(matmulOp.getB().getType());

    int64_t k_of_A = a_type.getShape()[1];
    int64_t k_of_B = b_type.getShape()[0];

    // Verify that the two K dimensions match; otherwise the matmul is invalid.
    if (k_of_A != k_of_B) {
      return rewriter.notifyMatchFailure(
          matmulOp,
          "K dimensions of A and B do not match");
    }

    // Verify K is divisible by the tile size; if not, we do not transform.
    if (k_of_A % TILE_SIZE != 0) {
      return rewriter.notifyMatchFailure(
          matmulOp,
          "K dimension not divisible by tile size");
    }

    // Create tiled reshape operations
    auto a_tiled = createATiledReshape(rewriter, matmulOp);
    auto b_tiled = createBTiledReshape(rewriter, matmulOp);

    // Create tiled matmul
    auto matmul_tiles =
        createTiledMatmul(rewriter, matmulOp, a_tiled, b_tiled);

    // Create reduce/sum operation
    auto reduceSum = createTiledReduce(rewriter, matmulOp, matmul_tiles);

    // Replace original matmul with final reduce
    rewriter.replaceOp(matmulOp, reduceSum.getResult());

    return success();
  }
};

struct TileMatmulPass : public impl::TileMatmulBase<TileMatmulPass> {
  using TileMatmulBase::TileMatmulBase;

  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<MatmulTilingPattern>(ctx);

    // Apply the patterns greedily. If the transformation fails for some reason
    // report pass failure so the pipeline can react.
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::transforms
