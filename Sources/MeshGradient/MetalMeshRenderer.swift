
import Foundation
import Metal
import MetalKit
@_implementationOnly import MeshGradientCHeaders

public final class MetalMeshRenderer: NSObject {

    private var commandQueue: MTLCommandQueue?
    private var bufferPool: MTLBufferPool?

    private var computeShuffleCoefficients: MTLComputeShuffleCoefficientsFunction?
    private var computeHermitPatchMatrix: MTLComputeHermitPatchMatrixFunction?
    private var computeMeshTrianglePrimitives: MTLComputeMeshTrianglePrimitivesFunction?
    private var drawMesh: MTLDrawMeshTrianglesFunction?
    
    private var viewportSize: vector_float2
    private let subdivisions: Int
    private let meshDataProvider: MeshDataProvider

    public init(
        metalKitView mtkView: MTKView?,
        meshDataProvider: MeshDataProvider,
        viewportSize: vector_float2 = .zero,
        subdivisions: Int = 18
    ) {
        self.viewportSize = viewportSize
        self.subdivisions = subdivisions
        self.meshDataProvider = meshDataProvider

        guard
            let device = mtkView?.device ?? MTLCreateSystemDefaultDevice(),
            let defaultLibrary = try? device.makeDefaultLibrary(bundle: .module)
        else {
            assertionFailure()
            return
        }

        let bufferPool = MTLBufferPool(device: device)

        do {
            computeShuffleCoefficients = try .init(
                device: device,
                library: defaultLibrary,
                bufferPool: bufferPool
            )

            computeHermitPatchMatrix = try .init(device: device, library: defaultLibrary)
            computeMeshTrianglePrimitives = try .init(device: device, library: defaultLibrary)

            drawMesh = try .init(
                device: device,
                library: defaultLibrary,
                pixelFormat: mtkView?.colorPixelFormat ?? .bgra8Unorm
            )
            
        } catch {
            assertionFailure(error.localizedDescription)
        }

        commandQueue = device.makeCommandQueue()
        self.bufferPool = bufferPool
    }

    // MARK: Public Methods

    public func draw(
        pixelFormat: MTLPixelFormat,
        renderPassDescriptor: MTLRenderPassDescriptor?,
        currentDrawable: CAMetalDrawable?,
        completion: ((MTLTexture?) -> Void)? = nil
    ) {

        guard
            let commandQueue = commandQueue,
            let commandBuffer = commandQueue.makeCommandBuffer()
        else {
            completion?(nil)
            return
        }

        let grid = meshDataProvider.grid

        guard
            let (resultBuffer, _, resultElementsCount) = calculateTriangles(
                grid: grid,
                subdivisions: subdivisions,
                commandBuffer: commandBuffer
            ),
            let drawMesh = self.drawMesh
        else {
            completion?(nil)
            return
        }

        if let renderPassDescriptor {
            drawMesh.call(
                meshVertices: resultBuffer,
                meshVerticesCount: resultElementsCount,
                renderPassDescriptor: renderPassDescriptor,
                commandBuffer: commandBuffer,
                viewportSize: viewportSize
            )
        } else {
            completion?(nil)
            return
        }

        if let currentDrawable {
            commandBuffer.present(currentDrawable)
        }

        if let texture = currentDrawable?.texture {
            commandBuffer.addCompletedHandler { _ in
                completion?(texture)
            }
        } else {
            completion?(nil)
            return
        }

        commandBuffer.commit()
    }
}

// MARK: MTKViewDelegate

extension MetalMeshRenderer: MTKViewDelegate {

    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        viewportSize.x = Float(size.width)
        viewportSize.y = Float(size.height)
    }

    public func draw(in view: MTKView) {
        draw(
            pixelFormat: view.colorPixelFormat,
            renderPassDescriptor: view.currentRenderPassDescriptor,
            currentDrawable: view.currentDrawable
        )
    }
}

// MARK: Private Methods

private extension MetalMeshRenderer {
	
    func calculateTriangles(
        grid: Grid<ControlPoint>,
        subdivisions: Int,
        commandBuffer: MTLCommandBuffer
    ) -> (buffer: MTLBuffer, length: Int, elementsCount: Int)? {

		let resultVerticesSize = getResultVerticesSize(grid: grid, subdivisions: subdivisions)
		let resultTrianglesSize = (resultVerticesSize.width - 1) * (resultVerticesSize.height - 1) * 6
		let resultTrianglesBufferSize = MemoryLayout<MeshVertex>.stride * resultTrianglesSize
		
		guard
            let (triangleStripBuf, _, triangleStripCount) = calculateMeshTriangles(
                grid: grid,
                subdivisions: subdivisions,
                commandBuffer: commandBuffer
            ),
            let resultTrianglesBuffer = bufferPool?[resultTrianglesBufferSize, .storageModePrivate],
            let computeMeshTrianglePrimitives = self.computeMeshTrianglePrimitives
		else {
            return nil
        }

		commandBuffer.addCompletedHandler { _ in
			self.bufferPool?[resultTrianglesBufferSize, .storageModePrivate] = resultTrianglesBuffer
		}
        
        computeMeshTrianglePrimitives.call(
            gridSize: resultVerticesSize,
            resultTrianglesBuffer: resultTrianglesBuffer,
            finalVertices: triangleStripBuf,
            finalVerticesSize: triangleStripCount,
            commandBuffer: commandBuffer
        )

        return (resultTrianglesBuffer, resultTrianglesBufferSize, resultTrianglesSize)
	}
	
	func getResultVerticesSize(
        grid: Grid<ControlPoint>,
        subdivisions: Int
    ) -> (width: Int, height: Int) {
		return (width: (grid.width - 1) * subdivisions, height: (grid.height - 1) * subdivisions)
	}
	
    func calculateMeshTriangles(
        grid: Grid<ControlPoint>,
        subdivisions: Int,
        commandBuffer: MTLCommandBuffer
    ) -> (buffer: MTLBuffer, length: Int, elementsCount: Int)? {

		let resultVerticesSize = getResultVerticesSize(grid: grid, subdivisions: subdivisions)
		
		let intermediateSize = (grid.width - 1) * (grid.height - 1)
		let intermediateBufferSize = intermediateSize * MemoryLayout<MeshIntermediateVertex>.stride
		
		let finalVerticesSize = resultVerticesSize.width * resultVerticesSize.height
		let finalVerticesBufferSize = MemoryLayout<MeshVertex>.stride * finalVerticesSize
		
        guard
            let intermediateResultBuffer = bufferPool?[intermediateBufferSize, .storageModePrivate],
            let finalVerticesBuffer = bufferPool?[finalVerticesBufferSize, .storageModePrivate],
            let computeShuffleCoefficients = self.computeShuffleCoefficients,
            let computeHermitPatchMatrix = self.computeHermitPatchMatrix
		else {
			assertionFailure()
			return nil
		}

		commandBuffer.addCompletedHandler { _ in
			self.bufferPool?[intermediateBufferSize, .storageModePrivate] = intermediateResultBuffer
			self.bufferPool?[finalVerticesBufferSize, .storageModePrivate] = finalVerticesBuffer
		}
		
		commandBuffer.label = "Show Mesh Buffer"
        
        computeShuffleCoefficients.call(
            grid: grid,
            intermediateResultBuffer: intermediateResultBuffer,
            commandBuffer: commandBuffer
        )
        
        computeHermitPatchMatrix.call(
            subdivisions: subdivisions,
            resultBuffer: finalVerticesBuffer,
            intermediateResultBuffer: intermediateResultBuffer,
            gridWidth: resultVerticesSize.width,
            intermediateResultBufferSize: intermediateSize,
            commandBuffer: commandBuffer
        )
        
        return (finalVerticesBuffer, finalVerticesBufferSize, finalVerticesSize)
	}
    
	func unwrap<Element>(buffer: MTLBuffer, length: Int? = nil, elementsCount: Int) -> [Element] {
		let rawPointer = buffer.contents()
		let length = length ?? MemoryLayout<Element>.stride * elementsCount
		let typedPointer = rawPointer.bindMemory(to: Element.self, capacity: length)
		let bufferedPointer = UnsafeBufferPointer(start: typedPointer, count: length)
		
		var result: [Element] = []
		for i in 0..<elementsCount {
			result.append(bufferedPointer[i])
		}
		return result
	}
}
