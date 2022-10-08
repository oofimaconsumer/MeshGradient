//
//  MeshGradientType.swift
//  
//
//  Created by Timur Ganiev on 08.10.2022.
//

import Foundation

public enum MeshGradientType {

    case animated(grid: Grid<ControlPoint>, configuration: MeshAnimator.Configuration)
    case `static`(grid: Grid<ControlPoint>)

    public func makeDataProvider() -> MeshDataProvider {
        switch self {
        case let .animated(grid, configuration):
            return MeshAnimator(grid: grid, configuration: configuration)

        case let .static(grid):
            return StaticMeshDataProvider(grid: grid)
        }
    }
}
