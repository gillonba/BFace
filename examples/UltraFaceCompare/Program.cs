using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using NcnnDotNet.OpenCV;
using System.IO;
using System.Linq;
using UltraFaceDotNet;

// See https://aka.ms/new-console-template for more information
public class Program {
    private static UltraFaceParameter param = new UltraFaceParameter() {
        BinFilePath = Path.Combine("assets", "models", "RFB-320.bin"),
        ParamFilePath = Path.Combine("assets", "models", "RFB-320.param"),
        InputWidth=320,
        InputLength = 240,
        NumThread = 1,
        ScoreThreshold = .7f
    };
    public static float Threshold => .05f;


    public static void Main() {
        System.Console.WriteLine("Hello, World!");

        //var bface = new BarronGillon.BFace.BFace(Path.Combine("assets", "models", "bface_uf.onnx"));
        var bface = new BarronGillon.BFace.BFace(Path.Combine("assets", "models", "bface.onnx"));
        var uface = UltraFace.Create(param);
        var mismatchoutdir = Path.Combine("assets", "out", "mismatch");

        //Purge the results from the prior run
        System.IO.Directory.Delete(mismatchoutdir, true);

        //Get the files to be compared
        var extBlacklist = new string[] {".json", ".webp"};
        var filedir = Path.Combine("assets", "images");
        var files = Directory.EnumerateFiles(filedir)
            .Where(x => !extBlacklist.Contains(Path.GetExtension(x)));

        //For each, get locations from both BFace and UltraFace
        int ctr = 1;
        int total = files.Count();
        int ctrMatch = 0;
        int ctrMismatch = 0;
        var swBFaceLocs = new System.Diagnostics.Stopwatch();
        var swUFaceLocs = new System.Diagnostics.Stopwatch();
        foreach (var f in files) {
            System.Console.WriteLine($"Testing {ctr} of {total} {f}");

            swBFaceLocs.Start();
            var img = new System.Drawing.Bitmap(f);
            var bfacelocs = bface.GetFaceLocations(img);
            swBFaceLocs.Stop();

            swUFaceLocs.Start();
            var bmp = new System.Drawing.Bitmap(f);
            using var frame = Cv2.ImDecode(bitmapToByteArray(bmp));
            using var inMat =
                NcnnDotNet.Mat.FromPixels(frame.Data, NcnnDotNet.PixelType.Bgr2Rgb, frame.Cols, frame.Rows);
            var ufacelocs = uface.Detect(inMat);
            swUFaceLocs.Stop();
                
            //If the counts don't match, we have a mismatch.
            var match = bfacelocs.Count() == ufacelocs.Count();

            //foreach location, if ANY dimensions are off by the threshold or more, we have a mismatch
            if(match) {
                foreach(var bl in bfacelocs) {
                    if (!ufacelocs.Any(x => IsNear(x, bl, bmp))) {
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
                var frdnimgout = Path.Combine(mismatchoutdir, "uface", Path.GetFileName(f));
                if (!Directory.Exists(Path.GetDirectoryName(frdnimgout))) Directory.CreateDirectory(Path.GetDirectoryName(frdnimgout));
                AnnotateImg(img, ufacelocs, frdnimgout);
                
                //Write FRDN -> YOLO
                //Writes the FRDN locations to a YOLO file so that we can use it for training future iterations of BFace
                var yoloout = Path.Combine(mismatchoutdir, "yolo", Path.ChangeExtension(Path.GetFileName(f), "txt"));
                if (!Directory.Exists(Path.GetDirectoryName(yoloout))) Directory.CreateDirectory(Path.GetDirectoryName(yoloout));
                var yololines = FRDNToYOLO(ufacelocs, bmp);
                File.WriteAllLines(yoloout, yololines);

                ctrMismatch++;
            }
            ctr++;
        }
        
        //Output the statistics
        System.Console.WriteLine($"Found {ctrMatch} matches and {ctrMismatch} mismatches");
        System.Console.WriteLine("Search times for BFace: {0} FRDN: {1}", swBFaceLocs.Elapsed, swUFaceLocs.Elapsed);
    }
    
    /// <summary>
    /// Borrowed from another project, should be re-written to not use System.Drawing
    /// </summary>
    /// <param name="source"></param>
    /// <param name="ifef"></param>
    /// <param name="outfile"></param>
    private static void AnnotateImg(System.Drawing.Bitmap source, 
        IEnumerable<FaceInfo> ifef, string outfile) {
     
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
                                int x = (int)fef.X1;
                                int y = (int)fef.Y1;
                                int width = (int)(fef.X2 - fef.X1);
                                int height = (int)(fef.Y2 - fef.Y1);
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

    private static byte[] bitmapToByteArray(System.Drawing.Bitmap bmp) {
        MemoryStream stream = new MemoryStream();
        bmp.Save(stream, ImageFormat.Png);
        return stream.ToArray();
    }

    public static bool IsNear(FaceInfo x, BarronGillon.BFace.Location y, Bitmap img) {
        float maxdifX = img.Width * Threshold;
        float maxdifY = img.Height * Threshold;



        if (System.Math.Abs(x.X1 - y.Left) > maxdifX) return false;
        if (System.Math.Abs(x.X2 - y.Right) > maxdifX) return false;
        if (System.Math.Abs(x.Y1 - y.Top) > maxdifY) return false;
        if (System.Math.Abs(x.Y2 - y.Bottom) > maxdifY) return false;

        return true;
    }

    public static IEnumerable<string> FRDNToYOLO(IEnumerable<FaceInfo> locs, 
        Bitmap img){
        var ret = new List<string>();

        foreach (var l in locs) {
            var centerX = ((l.X1 + l.X2) / 2) / img.Width;
            var centerY = ((l.Y1 + l.Y2) / 2) / img.Height;
            var width =  (l.X2 - l.X1) / img.Width;
            var height = (l.Y2 - l.Y1) / img.Height;
            
            ret.Add($"0 {centerX} {centerY} {width} {height}");
        }

        return ret;
    }
}