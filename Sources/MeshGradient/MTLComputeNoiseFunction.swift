import Foundation
import Metal
import simd
@_implementationOnly import MeshGradientCHeaders

final class MTLComputeNoiseFunction {

    private let device: MTLDevice
    private let pipelineState: MTLComputePipelineState
    
    init(device: MTLDevice, library: MTLLibrary) throws {
        
        guard
            let computeNoiseFunction = library.makeFunction(name: "computeNoise")
        else {
            throw MeshGradientError.metalFunctionNotFound(name: "computeNoise")
        }

        self.device = device
        self.pipelineState = try device.makeComputePipelineState(function: computeNoiseFunction)
    }
    
    func call(
        viewportSize: simd_float2,
        pixelFormat: MTLPixelFormat,
        commandQueue: MTLCommandQueue,
        uniforms: NoiseUniforms
    ) -> MTLTexture? {

        guard
            uniforms.noiseAlpha > .zero,
            let commandBuffer = commandQueue.makeCommandBuffer()
        else {
            return nil
        }

        let width = Int(viewportSize.x)
        let height = Int(viewportSize.y)


        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )

        textureDescriptor.usage = [
            .shaderRead,
            .shaderWrite,
            .renderTarget,
            .pixelFormatView
        ]

        guard
            let noiseTexture = device.makeTexture(descriptor: textureDescriptor),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            return nil
        }

        var uniforms = uniforms
        
        let threadgroupCounts = MTLSize(width: 8, height: 8, depth: 1)

        let threadgroups = MTLSize(
            width: noiseTexture.width / threadgroupCounts.width,
            height: noiseTexture.height / threadgroupCounts.height,
            depth: 1
        )
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setTexture(noiseTexture, index: Int(ComputeNoiseInputIndexOutputTexture.rawValue))

        encoder.setBytes(
            &uniforms,
            length: MemoryLayout.size(ofValue: uniforms),
            index: Int(ComputeNoiseInputIndexUniforms.rawValue)
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupCounts)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return noiseTexture
    }
}
