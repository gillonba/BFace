// See https://aka.ms/new-console-template for more information
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection.PortableExecutable;
using System.Runtime.CompilerServices;
using BarronGillon.BFace;
using FaceRecognitionDotNet;

public class Program {
    // YOLO commands:
    // train (from yolo dir):
    // python train.py --data data/bface.yaml
    // export: 
    // python export.py --weights runs/train/exp12/weights/best.pt --img 640 --include onnx


    public static float threshold => .01f;

    public static void Main() {
        System.Console.WriteLine("Hello, World!");

        var bface = new BarronGillon.BFace.BFace(Path.Combine("assets", "models", "bface.onnx"));
        var frdn = FaceRecognitionDotNet.FaceRecognition.Create(Path.Combine("assets", "models"));
        var mismatchoutdir = Path.Combine("assets", "out", "mismatch");
        
        //Purge the results from the prior run
        System.IO.Directory.Delete(mismatchoutdir, true);

        //Get the files to be compared
        var extBlacklist = new string[] {".json", ".webp"};
        var filedir = Path.Combine("assets", "images");
        var files = Directory.EnumerateFiles(filedir)
            .Where(x => !extBlacklist.Contains(Path.GetExtension(x)));

        //For each, get locations from both FRDN and BFace
        int ctr = 1;
        int total = files.Count();
        int ctrMatch = 0;
        int ctrMismatch = 0;
        var swBFaceLocs = new System.Diagnostics.Stopwatch();
        var swFRDNLocs = new System.Diagnostics.Stopwatch();
        foreach(var f in files) {
            System.Console.WriteLine($"Testing {ctr} of {total} {f}");            

            swBFaceLocs.Start();
            var img = new System.Drawing.Bitmap(f);
            var bfacelocs = bface.GetFaceLocations(img);
            swBFaceLocs.Stop();

            swFRDNLocs.Start();
            var frdnimg = FaceRecognition.LoadImageFile(f);
            var frdnlocs = frdn.FaceLocations(frdnimg);
            swFRDNLocs.Stop();
            
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
            if(match) {
                ctrMatch++;
            } else {
                //Write the source images
                var srcimg = System.IO.Path.Combine(mismatchoutdir, "src", System.IO.Path.GetFileName(f));
                if (!Directory.Exists(Path.GetDirectoryName(srcimg))) Directory.CreateDirectory(Path.GetDirectoryName(srcimg));
                System.IO.File.Copy(f, srcimg);
                
                //Write annotated BFace
                var bfaimg = Path.Combine(mismatchoutdir, "bface", Path.GetFileName(f));
                if (!Directory.Exists(Path.GetDirectoryName(bfaimg))) Directory.CreateDirectory(Path.GetDirectoryName(bfaimg));
                var annotated = bface.Annotate(img, bfacelocs);
                annotated.Save(bfaimg);
                
                //Write annotated FRDN
                var frdnimgout = Path.Combine(mismatchoutdir, "frdn", Path.GetFileName(f));
                if (!Directory.Exists(Path.GetDirectoryName(frdnimgout))) Directory.CreateDirectory(Path.GetDirectoryName(frdnimgout));
                AnnotateImg(img, frdnlocs, frdnimgout);
                
                //Write FRDN -> YOLO
                //Writes the FRDN locations to a YOLO file so that we can use it for training future iterations of BFace
                var yoloout = Path.Combine(mismatchoutdir, "yolo", Path.ChangeExtension(Path.GetFileName(f), "txt"));
                if (!Directory.Exists(Path.GetDirectoryName(yoloout))) Directory.CreateDirectory(Path.GetDirectoryName(yoloout));
                var yololines = FRDNToYOLO(frdnlocs, frdnimg);
                File.WriteAllLines(yoloout, yololines);

                ctrMismatch++;
            }
            ctr++;
        }

        //Output the statistics
        System.Console.WriteLine($"Found {ctrMatch} matches and {ctrMismatch} mismatches");
        System.Console.WriteLine("Search times for BFace: {0} FRDN: {1}", swBFaceLocs.Elapsed, swFRDNLocs.Elapsed);
    }

    /// <summary>
    /// Borrowed from another project, should be re-written to not use System.Drawing
    /// </summary>
    /// <param name="source"></param>
    /// <param name="ifef"></param>
    /// <param name="outfile"></param>
    public static void AnnotateImg(System.Drawing.Bitmap source, 
        IEnumerable<FaceRecognitionDotNet.Location> ifef, string outfile) {
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
    
    /// <summary>
    /// Takes the YOLO locations and outputs a string containing the YOLO annotations
    /// </summary>
    /// <param name="locs"></param>
    /// <remarks>The YOLO annotations go to a text file with the same name as the image.  Each line in the text file
    /// contains the data for one annotation.  The format for each line is:
    /// [object-class] [center_x] [center_y] [width] [height]
    /// This model is a single-class model (for now)
    /// The dimensions are NOT pixel locations, they are normalized 0-1
    /// </remarks>
    /// <returns></returns>
    public static IEnumerable<string> FRDNToYOLO(IEnumerable<FaceRecognitionDotNet.Location> locs, 
    FaceRecognitionDotNet.Image img){
        var ret = new List<string>();

        foreach (var l in locs) {
            var centerX = ((float)(l.Left + l.Right) / 2) / img.Width;
            var centerY = ((float)(l.Top + l.Bottom) / 2) / img.Height;
            var width = ((float) (l.Right - l.Left)) / img.Width;
            var height = ((float) (l.Bottom - l.Top)) / img.Height;
            
            ret.Add($"0 {centerX} {centerY} {width} {height}");
        }

        return ret;
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

