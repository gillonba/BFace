using System;
using BarronGillon.BFace.YOLO;
using Microsoft.ML;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Processing;
//using Color = SixLabors.ImageSharp.Color;
using Image = SixLabors.ImageSharp.Image;


namespace BarronGillon.BFace {

    public class BFace {
        private static readonly string[] classNames = new []{"face"};
        //static readonly string[] classNames = new string[] { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
        //private static readonly string[] classNames = new string[] {"leaf", "flower", "fruit"};
        
        //private readonly PredictionEngine<YOLOv5_BitmapData, YOLOv5_Prediction> _predictionEngine;
        private readonly BarronGillon.BFace.YOLOv5NET.YoloScorer<YOLOv5NET.Models.YoloBFaceP5Model> _scorer;
        private readonly float _threshold;

        public BFace(string modelFile, float threshold = 0) {
            if (modelFile == null) throw new ArgumentNullException(modelFile);
            if(!File.Exists(modelFile)) throw new ArgumentException("Could not find modelPath " + modelFile, nameof(modelFile));

            _threshold = threshold;
            
            _scorer = new YOLOv5NET.YoloScorer<YOLOv5NET.Models.YoloBFaceP5Model>(modelFile);
        }
        

        /// <summary>
        /// 
        /// </summary>
        /// <param name="image"></param>
        /// <param name="locations"></param>
        /// <returns></returns>
        [System.Obsolete]
        public Bitmap Annotate(Bitmap image, IEnumerable<Location> locations) {
            var img = BitmapToImage(image);
            
            img.Mutate(x => {
                var font = SixLabors.Fonts.SystemFonts.CreateFont("Ubuntu", 18);

                //SixLabors.ImageSharp.Color.Red
                foreach (var res in locations) {
                    var rect = new SixLabors.ImageSharp.Rectangle(res.Left, res.Top, res.Right - res.Left,
                        res.Bottom - res.Top);
                    
                    x.Draw(SixLabors.ImageSharp.Color.Red, 1, rect);

                    string txt = res.Confidence.ToString("0.00");
                    var loc = new SixLabors.ImageSharp.PointF(res.Left, res.Top);
                    x.DrawText(txt, font, SixLabors.ImageSharp.Color.Blue, loc);
                }
            });

            var ret = ImageToBitmap(img);
            return ret;
        }

        public Image Annotate(Image img, IEnumerable<Location> locations) {

            img.Clone(x => {
                var font = SixLabors.Fonts.SystemFonts.CreateFont("Ubuntu", 18);

                //SixLabors.ImageSharp.Color.Red
                foreach (var res in locations) {
                    var rect = new SixLabors.ImageSharp.Rectangle(res.Left, res.Top, res.Right - res.Left,
                        res.Bottom - res.Top);
                    
                    x.Draw(SixLabors.ImageSharp.Color.Red, 1, rect);

                    string txt = res.Confidence.ToString("0.00");
                    var loc = new SixLabors.ImageSharp.PointF(res.Left, res.Top);
                    x.DrawText(txt, font, SixLabors.ImageSharp.Color.Blue, loc);
                }
            });

            return img;
        }
        
        public IEnumerable<Location> GetFaceLocations(Bitmap image) {
            List<YOLOv5NET.YoloPrediction> predictions = _scorer.Predict(image);
            
            var r = predictions.Select(x => new Location() {
                Top = (int)x.Rectangle.Top,
                Bottom = (int)x.Rectangle.Bottom,
                Left = (int)x.Rectangle.Left,
                Right = (int)x.Rectangle.Right,
                Confidence = x.Score
            });
            return r;
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

        private System.Drawing.Bitmap ImageToBitmap(SixLabors.ImageSharp.Image img) {
            System.Drawing.Bitmap ret = null;
            using (var ms = new MemoryStream()) {
                var imgEncoder = img.GetConfiguration().ImageFormatsManager.FindEncoder(PngFormat.Instance);
                img.Save(ms, imgEncoder);
                ms.Seek(0, SeekOrigin.Begin);
                ret = new System.Drawing.Bitmap(ms);
            }

            return ret;
        }
    }
}