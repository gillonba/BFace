using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;

public class Program {
    public static void Main() {
        var inputDir = Path.Combine("assets", "images");
        var outputDir = Path.Combine("assets", "out");
        var modelPath = System.IO.Path.Combine("assets", "models", "bface.onnx");
        
        var detector = new BarronGillon.BFace.BFace(modelPath);

        var inputFiles = Directory.GetFiles(inputDir);
        foreach(var f in inputFiles) {
            System.Console.WriteLine("Checking " + f);
            
            // Get the results
            var img = new Bitmap(f);
            Stopwatch swDetect = Stopwatch.StartNew();
            var results = detector.GetFaceLocations(img);
            swDetect.Stop();
            System.Console.WriteLine("Detected in " + swDetect.Elapsed.ToString());
            
            // Output the results to console
            System.Console.WriteLine("Found {0} results:", results.Count());
            foreach (var r in results) {
                System.Console.WriteLine("Left: {0} Top: {1} Right: {2} Bottom: {3} Confidence: {4}, ratio: {5}, imgratio: {6}",
                    r.Left, r.Top, r.Right, r.Bottom, r.Confidence, (float)(r.Right - r.Left) / (r.Bottom - r.Top), (float)img.Width / img.Height);
            }

            // Write the results to an image
            var imgPathOut = Path.Combine(outputDir, Path.GetFileName(f));
            if (!Directory.Exists(Path.GetDirectoryName(imgPathOut)))
                Directory.CreateDirectory(Path.GetDirectoryName(imgPathOut));
            img = detector.Annotate(img, results);
            img.Save(imgPathOut);
            System.Console.WriteLine("Wrote results to " + imgPathOut);
            System.Console.WriteLine();
        }
    }
}