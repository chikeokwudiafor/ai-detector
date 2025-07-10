
import Foundation
import UIKit

public class AIDetectorSDK {
    private let baseURL: String
    private let session = URLSession.shared
    
    public init(baseURL: String = "https://your-repl-name.your-username.replit.app") {
        self.baseURL = baseURL
    }
    
    public struct DetectionResponse: Codable {
        let success: Bool
        let result: String?
        let confidence: Double?
        let resultType: String?
        let resultClass: String?
        let resultIcon: String?
        let resultDescription: String?
        let resultFooter: String?
        let error: String?
        
        enum CodingKeys: String, CodingKey {
            case success, result, confidence, error
            case resultType = "result_type"
            case resultClass = "result_class"
            case resultIcon = "result_icon"
            case resultDescription = "result_description"
            case resultFooter = "result_footer"
        }
    }
    
    public enum DetectionResult {
        case success(DetectionResponse)
        case error(String)
    }
    
    public func detectImage(_ image: UIImage, completion: @escaping (DetectionResult) -> Void) {
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            completion(.error("Failed to convert image to data"))
            return
        }
        
        let url = URL(string: "\(baseURL)/api/detect")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        performRequest(request, completion: completion)
    }
    
    public func detectText(_ text: String, completion: @escaping (DetectionResult) -> Void) {
        let url = URL(string: "\(baseURL)/api/detect")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody = ["text_content": text]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            completion(.error("Failed to encode request"))
            return
        }
        
        performRequest(request, completion: completion)
    }
    
    private func performRequest(_ request: URLRequest, completion: @escaping (DetectionResult) -> Void) {
        session.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    completion(.error(error.localizedDescription))
                    return
                }
                
                guard let data = data else {
                    completion(.error("No data received"))
                    return
                }
                
                do {
                    let detectionResponse = try JSONDecoder().decode(DetectionResponse.self, from: data)
                    completion(.success(detectionResponse))
                } catch {
                    completion(.error("Failed to parse response: \(error.localizedDescription)"))
                }
            }
        }.resume()
    }
}

// Usage example:
public class AIDetectorViewController: UIViewController {
    private let aiDetector = AIDetectorSDK()
    
    public func analyzeImage(_ image: UIImage) {
        aiDetector.detectImage(image) { result in
            switch result {
            case .success(let response):
                if response.success {
                    print("Result: \(response.result ?? "Unknown")")
                    print("Confidence: \(response.confidence ?? 0.0)")
                    print("Description: \(response.resultDescription ?? "")")
                } else {
                    print("Error: \(response.error ?? "Unknown error")")
                }
            case .error(let error):
                print("Network error: \(error)")
            }
        }
    }
}
