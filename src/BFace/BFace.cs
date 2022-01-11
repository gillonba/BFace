using System;
using BarronGillon.BFace.YOLO;
using Microsoft.ML;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Formats.Bmp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Processing;
using Color = SixLabors.ImageSharp.Color;
using Image = SixLabors.ImageSharp.Image;


namespace BarronGillon.BFace {

    public class BFace {
        private static readonly string[] classNames = new []{"face"};
        
        private readonly PredictionEngine<YOLOv5_BitmapData, YOLOv5_Prediction> _predictionEngine;

        public BFace(string modelPath) {
            if (modelPath == null) throw new ArgumentNullException(modelPath);
            if(!File.Exists(modelPath)) throw new ArgumentException("Could not find modelPath " + modelPath, nameof(modelPath));
            
            MLContext mlContext = new MLContext();

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "images", imageWidth: 640, imageHeight: 640, resizing: ResizingKind.Fill)
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "images", scaleImage: 1f / 255f, interleavePixelColors: false))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    /*shapeDictionary: new System.Collections.Generic.Dictionary<string, int[]>()
                    {
                        { "images", new[] { 1, 3, 640, 640 } },
                        { "output", new[] { 1, 25200, 85 } },
                    }*/
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

            /*using (var g = Graphics.FromImage(image)) {
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
            }*/

            //System.IO.MemoryStream ms = new MemoryStream();
            //image.Save(ms,ImageFormat.Bmp);
            //var img = SixLabors.ImageSharp.Image.Load(ms);
            SixLabors.ImageSharp.Image img = null;
            using (var ms = new MemoryStream()) {
                image.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
                ms.Seek(0, SeekOrigin.Begin);
                img = Image.Load(ms);
            }
            
            img.Mutate(x => {
                //SixLabors.ImageSharp.Color.Red
                foreach (var res in locations) {
                    var rect = new SixLabors.ImageSharp.Rectangle(res.Left, res.Top, res.Right - res.Left,
                        res.Bottom - res.Top);
                    
                    x.Draw(SixLabors.ImageSharp.Color.Red, 1, rect);
                    
                    
                    
                    //x.Fill(SixLabors.ImageSharp.Drawing.)
                    //var font = SixLabors.Fonts.SystemFonts.CreateFont("Arial", 12); //new SixLabors.Fonts.Font("" ,2);
                    //var font = SixLabors.Fonts.SystemFonts.Find("sansserif");
                    var font = SixLabors.Fonts.SystemFonts.CreateFont("Ubuntu", 18);
                    //x.DrawText(res.Confidence.ToString("0.00"), font, Color.Blue, new PointF(res.Left, res.Top));
                    string txt = res.Confidence.ToString("0.00");
                    var loc = new SixLabors.ImageSharp.PointF(res.Left, res.Top);
                    x.DrawText(txt, font, Color.Blue, loc);
                }
            });

            //System.IO.MemoryStream mso = new MemoryStream();
            //img.Save(mso, new BmpEncoder());
            //Bitmap ret = new Bitmap(mso);
            //return ret;
            System.Drawing.Bitmap ret = null;
            using (var ms = new MemoryStream()) {
                var imgEncoder = img.GetConfiguration().ImageFormatsManager.FindEncoder(PngFormat.Instance);
                img.Save(ms, imgEncoder);
                ms.Seek(0, SeekOrigin.Begin);
                ret = new System.Drawing.Bitmap(ms);
            }

            return ret;
        }

        public IEnumerable<Location> GetFaceLocations(Bitmap image) {
            return GetFaceLocations(new[] {image}).First();
        }
        
        /// <summary>
        /// Get the locations of the faces in a batch of images
        /// </summary>
        /// <param name="images">The images to scan.  For now, ML.Net does not officially support ImageSharp, though it is on their roadmap</param>
        /// <returns></returns>
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
                    //var predict = _predictionEngine.Predict(new YOLOv5_BitmapData() {Image = BitmapToImage(bitmap)});
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

        private SixLabors.ImageSharp.Image BitmapToImage(System.Drawing.Bitmap bmp) {
            SixLabors.ImageSharp.Image img = null;
            using (var ms = new MemoryStream()) {
                bmp.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
                ms.Seek(0, SeekOrigin.Begin);
                img = Image.Load(ms);
            }

            return img;
        }
    }
}