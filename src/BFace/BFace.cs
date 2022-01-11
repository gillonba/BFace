using System;
using BarronGillon.BFace.YOLO;
using Microsoft.ML;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;

namespace BarronGillon.BFace {

    public class BFace {
        private static readonly string[] classNames = new []{"face"};

        const string imageFolder = @"Assets/Images";

        const string imageOutputFolder = @"Assets/Output";

        private readonly PredictionEngine<YOLOv5_BitmapData, YOLOv5_Prediction> _predictionEngine;

        public BFace(string modelPath) {
            if (modelPath == null) throw new ArgumentNullException(modelPath);
            if(!File.Exists(modelPath)) throw new ArgumentException("Could not find modelPath " + modelPath, nameof(modelPath));
            
            MLContext mlContext = new MLContext();

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "images", imageWidth: 640, imageHeight: 640, resizing: ResizingKind.Fill)
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "images", scaleImage: 1f / 255f, interleavePixelColors: false))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    /*shapeDictionary: new Dictionary<string, int[]>()
                    {
                        { "images", new[] { 1, 3, 640, 640 } },
                        { "output", new[] { 1, 25200, 85 } },
                    },*/
                    inputColumnNames: new[]
                    {
                        "images"
                    },
                    outputColumnNames: new[]
                    {
                        "output"
                    },
                    modelFile: modelPath));

            // Fit on empty list to obtain input data schema
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<YOLOv5_BitmapData>()));

            // Create prediction engine
            _predictionEngine = mlContext.Model.CreatePredictionEngine<YOLOv5_BitmapData, YOLOv5_Prediction>(model);
        }
        
        private void fmain() {
            System.Console.WriteLine("Hello World!");

            string modelPath = "";
            
            Directory.CreateDirectory(imageOutputFolder);
            MLContext mlContext = new MLContext();

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "images", imageWidth: 640, imageHeight: 640, resizing: ResizingKind.Fill)
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "images", scaleImage: 1f / 255f, interleavePixelColors: false))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    /*shapeDictionary: new Dictionary<string, int[]>()
                    {
                        { "images", new[] { 1, 3, 640, 640 } },
                        { "output", new[] { 1, 25200, 85 } },
                    },*/
                    inputColumnNames: new[]
                    {
                        "images"
                    },
                    outputColumnNames: new[]
                    {
                        "output"
                    },
                    modelFile: modelPath));

            // Fit on empty list to obtain input data schema
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<YOLOv5_BitmapData>()));

            // Create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<YOLOv5_BitmapData, YOLOv5_Prediction>(model);
            //var predictionEngine = mlContext.Model.CreatePredictionEngine<YoloV5BitmapData, YoloV5l_Prediction>(model);

            // save model
            //mlContext.Model.Save(model, predictionEngine.OutputSchema, Path.ChangeExtension(modelPath, "zip"));

            var inputFiles = System.IO.Directory.GetFiles(imageFolder).Select(x => System.IO.Path.GetFileName(x));

            var timer = Stopwatch.StartNew();
            foreach(string imageName in inputFiles)  //foreach (string imageName in new string[] { "kite.jpg", "kite_416.jpg", "dog_cat.jpg", "cars road.jpg", "ski.jpg", "ski2.jpg" })
            {
                System.Console.WriteLine($"Testing {imageName}...");
                using(var bitmap = new Bitmap(Image.FromFile(Path.Combine(imageFolder, imageName)))) //using (var bitmap = new Bitmap(Image.FromFile(Path.Combine(imageFolder, imageName))))
                {
                    // predict
                    var predict = predictionEngine.Predict(new YOLOv5_BitmapData() { Image = bitmap });
                    var results = predict.GetResults(classNames, 0.3f, 0.7f);

                    using (var g = Graphics.FromImage(bitmap))
                    {
                        foreach (var res in results)
                        {
                            // draw predictions
                            var x1 = res.BBox[0];
                            var y1 = res.BBox[1];
                            var x2 = res.BBox[2];
                            var y2 = res.BBox[3];
                            g.DrawRectangle(Pens.Red, x1, y1, x2 - x1, y2 - y1);
                            using (var brushes = new SolidBrush(Color.FromArgb(50, Color.Red)))
                            {
                                g.FillRectangle(brushes, x1, y1, x2 - x1, y2 - y1);
                            }

                            g.DrawString(res.Label + " " + res.Confidence.ToString("0.00"),
                                         new Font("Arial", 12), Brushes.Blue, new PointF(x1, y1));
                        }

                        string filename = Path.Combine(imageOutputFolder, Path.ChangeExtension(imageName, "_processed" + Path.GetExtension(imageName)));
                        bitmap.Save(filename);
                    }
                }
            }
            timer.Stop();

            System.Console.WriteLine(string.Format("Ran {0} in {1} avg {2} seconds", inputFiles.Count(),
                timer.Elapsed.ToString(), timer.Elapsed.TotalSeconds / inputFiles.Count()));
            System.Console.WriteLine("GoodBye World!");
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="image"></param>
        /// <param name="locations"></param>
        /// <returns></returns>
        public Bitmap Annotate(Bitmap image, IEnumerable<Location> locations) {
            // TODO: System.Drawing is not longer supported cross-platform!  To annotate, pick another library.  Currently it appears microsot suggests three options:
            // * ImageSharp
            // * SkiaSharp
            // * Microsoft.Maui.Graphics

            using (var g = Graphics.FromImage(image)) {
                foreach (var res in locations) {
                    // draw predictions
                    var x1 = res.Left;
                    var y1 = res.Top;
                    var x2 = res.Right;
                    var y2 = res.Bottom;
                    g.DrawRectangle(Pens.Red, x1, y1, x2 - x1, y2 - y1);
                    using (var brushes = new SolidBrush(Color.FromArgb(50, Color.Red))) {
                        g.FillRectangle(brushes, x1, y1, x2 - x1, y2 - y1);
                    }

                    g.DrawString(res.Confidence.ToString("0.00"),
                        new Font("Arial", 12), Brushes.Blue, new PointF(x1, y1));
                }

                return image;
            }
        }

        public IEnumerable<Location> GetFaceLocations(Bitmap image) {
            return GetFaceLocations(new[] {image}).First();
        }
        
        public IEnumerable<IEnumerable<Location>> GetFaceLocations(IEnumerable<Bitmap> images) {
            var timer = Stopwatch.StartNew();
            //foreach(string imageName in inputFiles)  //foreach (string imageName in new string[] { "kite.jpg", "kite_416.jpg", "dog_cat.jpg", "cars road.jpg", "ski.jpg", "ski2.jpg" })
            //{
                //System.Console.WriteLine($"Testing {imageName}...");
                //using(var bitmap = new Bitmap(Image.FromFile(Path.Combine(imageFolder, imageName)))) //using (var bitmap = new Bitmap(Image.FromFile(Path.Combine(imageFolder, imageName))))

                var ret = new List<IEnumerable<Location>>();
                
                foreach (var bitmap in images) {
                    // predict
                    var predict = _predictionEngine.Predict(new YOLOv5_BitmapData() {Image = bitmap});
                    var results = predict.GetResults(classNames, 0.3f, 0.7f);

                    var returnable = results.Select(x => new Location() {
                        Left = (int) x.BBox[0],
                        Right = (int) x.BBox[2],
                        Top = (int) x.BBox[1],
                        Bottom = (int) x.BBox[3],
                        Confidence = x.Confidence
                    });

                    ret.Add(returnable);

                    /*using (var g = Graphics.FromImage(bitmap))
                    {
                        foreach (var res in results)
                        {
                            // draw predictions
                            var x1 = res.BBox[0];
                            var y1 = res.BBox[1];
                            var x2 = res.BBox[2];
                            var y2 = res.BBox[3];
                            g.DrawRectangle(Pens.Red, x1, y1, x2 - x1, y2 - y1);
                            using (var brushes = new SolidBrush(Color.FromArgb(50, Color.Red)))
                            {
                                g.FillRectangle(brushes, x1, y1, x2 - x1, y2 - y1);
                            }

                            g.DrawString(res.Label + " " + res.Confidence.ToString("0.00"),
                                new Font("Arial", 12), Brushes.Blue, new PointF(x1, y1));
                        }

                        string filename = Path.Combine(imageOutputFolder, Path.ChangeExtension(imageName, "_processed" + Path.GetExtension(imageName)));
                        bitmap.Save(filename);
                    }*/

                }

                //}
            timer.Stop();
            
            //throw new NotImplementedException();
            return ret;
        }
    }
}