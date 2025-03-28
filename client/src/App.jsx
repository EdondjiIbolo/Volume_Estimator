import { useState } from "react";
import { Upload, Input, Button, message, Card, Image } from "antd";
import { UploadOutlined, DeleteOutlined } from "@ant-design/icons";
import axios from "axios";
import "./App.css"; // Import styles

function App() {
  const [fileList, setFileList] = useState([]);
  const [zoom, setZoom] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Function to convert milliliters to other units
  const convertMl = (ml) => ({
    milliliters: ml,
    cubicCentimeters: ml,
    cubicMeters: ml / 1_000_000,
    grams: ml,
  });
  const getUnit = (key) => {
    const units = {
      milliliters: "ml",
      cubicCentimeters: "cm³",
      cubicMeters: "m³",
      grams: "g",
    };
    return units[key] || "";
  };
  // Handle image upload
  const handleUploadChange = ({ fileList }) => setFileList(fileList);
  const handleRemove = () => setFileList([]);

  // Calculate volume based on backend response
  const calculateVolume = (data) => {
    const converted = convertMl(data.data);
    return Object.fromEntries(
      Object.entries(converted).map(([key, value]) => [key, value * 2])
    );
  };

  // Send image to Flask backend
  const handleSubmit = async () => {
    if (fileList.length === 0 || !zoom) {
      message.error("Please select an image and enter a zoom level");
      return;
    }

    const formData = new FormData();
    formData.append("image", fileList[0].originFileObj);
    formData.append("zoom", zoom);

    try {
      setLoading(true);
      const { data } = await axios.post("/predict", formData);
      setResult(calculateVolume(data));
      message.success("Prediction completed successfully");
    } catch (error) {
      message.error("Error predicting volume");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <Card title="Volume Estimation" bordered={false} className="card">
        {/* Image Upload */}
        {fileList.length === 0 ? (
          <Upload
            className="upload-box"
            listType="picture-card"
            fileList={fileList}
            onChange={handleUploadChange}
            beforeUpload={() => false}
          >
            <UploadOutlined />
          </Upload>
        ) : (
          <div className="preview-container">
            <Image
              className="preview-image"
              src={URL.createObjectURL(fileList[0].originFileObj)}
              alt="Preview"
            />
            <Button
              icon={<DeleteOutlined />}
              className="delete-btn"
              onClick={handleRemove}
            >
              Delete Image
            </Button>
          </div>
        )}

        {/* Zoom Input */}
        <Input
          type="number"
          placeholder="Zoom or scale level"
          value={zoom}
          onChange={(e) => setZoom(e.target.value)}
          className="input-zoom"
          suffix={"mm"}
        />

        {/* Submit Button */}
        <Button
          type="primary"
          onClick={handleSubmit}
          loading={loading}
          className="submit-btn"
        >
          Estimate Volume
        </Button>

        {/* Result */}
        {result && (
          <Card className="result-card">
            <h3>Estimated Volume:</h3>
            <ul>
              {Object.entries(result).map(([key, value]) => (
                <li key={key}>
                  <strong>
                    {key.replace(/([A-Z])/g, " $1").toLowerCase()}:
                  </strong>{" "}
                  {value} {getUnit(key)}
                </li>
              ))}
            </ul>
          </Card>
        )}
      </Card>
    </div>
  );
}

export default App;
