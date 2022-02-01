namespace BarronGillon.BFace.YOLOv5NET.Extensions {
    public static class RectangleExtensions
    {
        public static float Area(this System.Drawing.RectangleF source)
        {
            return source.Width * source.Height;
        }
    }
}