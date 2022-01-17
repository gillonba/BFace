// See https://aka.ms/new-console-template for more information
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using BarronGillon.BFace;
using FaceRecognitionDotNet;

public class Program {
        public static float threshold => .01f;

    public static void Main() {
        
        System.Console.WriteLine("Hello, World!");

        var bface = new BarronGillon.BFace.BFace(Path.Combine("assets", "models"));
        var frdn = FaceRecognitionDotNet.FaceRecognition.Create(Path.Combine("assets", "models"));
        var mismatchoutdir = Path.Combine("assets", "out", "mismatch");

        //Get the files to be compared
        var filedir = Path.Combine("assets", "images");
        var files = Directory.EnumerateFiles(filedir);

        //For each, get locations from both FRDN and BFace
        int ctr = 1;
        foreach(var f in files) {
            System.Console.WriteLine($"Testing {ctr} {f}");            
        
            var img = new System.Drawing.Bitmap(f);
            var bfacelocs = bface.GetFaceLocations(img);

            var frdnimg = FaceRecognition.LoadImageFile(f);
            var frdnlocs = frdn.FaceLocations(frdnimg);
            
            //If the counts don't match, we have a mismatch.
            var match = bfacelocs.Count() == frdnlocs.Count();

            //foreach location, if ANY dimensions are off by the threshold or more, we have a mismatch
            if(match) {
                foreach(var bl in bfacelocs) {
                    if (!frdnlocs.Any(x => IsNear(x, bl, frdnimg))) {
                        match = false;
                        break;
                    }
                }
            }

            //For each mismatch, copy the source image to a separate output directory along with yolo encodings from the
            //FRDN locations.  Also copy an annotated copy of the source file to another directory, so we can make sure we
            //aren't training on bad data
            if(match){
            
            } else {
                //Write the source images
                var srcimg = System.IO.Path.Combine(mismatchoutdir, "src", System.IO.Path.GetFileName(f));
                System.IO.File.Copy(f, srcimg);
                
                //Write annotated BFace
                var bfaimg = Path.Combine(mismatchoutdir, "bface", Path.GetFileName(f));
                var annotated = bface.Annotate(img, bfacelocs);
                annotated.Save(bfaimg);
                
                //Write annotated FRDN
                var frdnimgout = Path.Combine(mismatchoutdir, "frdn", Path.GetFileName(f));
                AnnotateImg(img, frdnlocs, frdnimgout);
                
            }
            ctr++;
        }

        //Output the statistics
    }

    /// <summary>
    /// Borrowed from another project, should be re-written to not use System.Drawing
    /// </summary>
    /// <param name="source"></param>
    /// <param name="ifef"></param>
    /// <param name="outfile"></param>
            public static void AnnotateImg(System.Drawing.Bitmap source, IEnumerable<FaceRecognitionDotNet.Location> ifef, string outfile) {
                                var c = System.Drawing.Color.Red;
                                 Brush b = new SolidBrush(c);
                                 Pen p = new Pen(c);
  
    
                var pixelFormats = new [] {System.Drawing.Imaging.PixelFormat.Undefined, System.Drawing.Imaging.PixelFormat.DontCare, 
                    System.Drawing.Imaging.PixelFormat.Format1bppIndexed, System.Drawing.Imaging.PixelFormat.Format4bppIndexed, System.Drawing.Imaging.PixelFormat.Format8bppIndexed, 
                    System.Drawing.Imaging.PixelFormat.Format16bppGrayScale, System.Drawing.Imaging.PixelFormat.Format16bppArgb1555};
    
                var tagFont = System.Drawing.SystemFonts.CaptionFont;
    
                if (ifef.Any()) {
                    using(System.Drawing.Image img = source){
                        System.Drawing.Graphics g = null;
                        System.Drawing.Bitmap tbmp = null;
                        if (pixelFormats.Contains(img.PixelFormat)) {
                            tbmp = new System.Drawing.Bitmap(img);
                            
                            g = System.Drawing.Graphics.FromImage(tbmp);
                        } else {
                            g = System.Drawing.Graphics.FromImage(img);
                        }
                        //using (var p = new Pen(Color.Red)) {
                            foreach (var fef in ifef) {
    
                                int x = fef.Left;
                                int y = fef.Top;
                                int width = fef.Right - fef.Left;
                                int height = fef.Bottom - fef.Top;
                                g.DrawRectangle(p, x, y, width, height);
                            }
                        //}
                        g.Dispose();
    
                        if (pixelFormats.Contains(img.PixelFormat)) {
                            tbmp.Save(outfile, System.Drawing.Imaging.ImageFormat.Png);
                        }
                        else {
                            img.Save(outfile, System.Drawing.Imaging.ImageFormat.Png);
                        }
                    }
                }
            }
    
    public static bool IsNear(FaceRecognitionDotNet.Location x, BarronGillon.BFace.Location y, FaceRecognitionDotNet.Image img) {
        float maxdifX = img.Width * threshold;
        float maxdifY = img.Height * threshold;
        
        if(System.Math.Abs(x.Left - y.Left) > maxdifX) return false;
        if(System.Math.Abs(x.Right - y.Right) > maxdifX) return false;
        if(System.Math.Abs(x.Top - y.Top) > maxdifY) return false;
        if(System.Math.Abs(x.Bottom - y.Bottom) > maxdifY) return false;

        return true;
    }
}

