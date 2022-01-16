// See https://aka.ms/new-console-template for more information

public class Program {
    public static void Main() {
        float threshold = .01f;
        
        System.Console.WriteLine("Hello, World!");
        
        //Get the files to be compared
        
        //For each, get locations from both FRDN and BFace
        
        //If the counts don't match, we have a mismatch.
        
        //foreach location, if ANY dimensions are off by the threshold or more, we have a mismatch
        
        //For each mismatch, copy the source image to a separate output directory along with yolo encodings from the
        //FRDN locations.  Also copy an annotated copy of the source file to another directory, so we can make sure we
        //aren't training on bad data
        
        //Output the statistics
    }
}

