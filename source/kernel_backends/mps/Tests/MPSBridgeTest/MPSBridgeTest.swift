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

    func test_fsin () {
        let x = UnsafeMutablePointer<Float>.allocate(capacity: 100)
        let y = UnsafeMutablePointer<Float>.allocate(capacity: 100)

        let _ = fsin_on_mps(input: x, output: y, count: 100)
    }
    
    static var allTests = [
      ("test_mps_2dfgemm", test_mps_2dfgemm),
      ("test_fsin", test_fsin)
    ]
}
