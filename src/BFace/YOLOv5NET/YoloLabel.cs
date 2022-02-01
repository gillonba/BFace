namespace BarronGillon.BFace.YOLOv5NET {
    /// <summary>
    /// Label of detected object.
    /// </summary>
    public class YoloLabel
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public YoloLabelKind Kind { get; set; }
        public System.Drawing.Color Color { get; set; }

        public YoloLabel()
        {
            this.Color = System.Drawing.Color.Yellow;
        }
    }
}