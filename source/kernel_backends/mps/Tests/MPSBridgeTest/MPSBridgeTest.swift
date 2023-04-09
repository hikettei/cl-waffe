import XCTest
@testable import MPSBridge

final class MPSBridgeTests: XCTestCase {

    func test_mps_2dfgemm () {
        let x = UnsafeMutablePointer<Float>.allocate(capacity: 100)
        let y = UnsafeMutablePointer<Float>.allocate(capacity: 100)
        let z = UnsafeMutablePointer<Float>.allocate(capacity: 100)

        let _ = mps_2dfgemm(alpha: 1.0, a: x, b: y, beta: 1.0, c: z,
                            m: 10, n: 10, k: 10, Transpose_A: false, Transpose_B: false)
        x.deallocate()
        y.deallocate()
        z.deallocate()

        XCTAssertEqual("", "")
    }
    
    func testExample() {
        XCTAssertEqual("Hello, World!", "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
