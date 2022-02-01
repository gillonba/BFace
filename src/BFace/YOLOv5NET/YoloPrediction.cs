namespace BarronGillon.BFace.YOLOv5NET {
    /// <summary>
    /// Object prediction.
    /// </summary>
    public class YoloPrediction
    {
        public YoloLabel Label { get; set; }
        public System.Drawing.RectangleF Rectangle { get; set; }
        public float Score { get; set; }

        public YoloPrediction() { }

        public YoloPrediction(YoloLabel label, float confidence) : this(label)
        {
            Score = confidence;
        }

        public YoloPrediction(YoloLabel label)
        {
            Label = label;
        }
    }
}