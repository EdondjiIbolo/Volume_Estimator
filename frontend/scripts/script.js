function uploadImage() {
  let fileInput = document.getElementById("imageInput");
  let zoomLevel = document.getElementById("zoomLevel").value;

  if (!fileInput.files.length || !zoomLevel) {
    alert("Please select an image and enter the zoom level.");
    return;
  }

  let formData = new FormData();
  formData.append("image", fileInput.files[0]);
  formData.append("zoom", zoomLevel);

  fetch("/predict", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("result").innerText =
        "Estimated Volume: " + data.volume + " cm³";
    })
    .catch((error) => console.error("Error:", error));
}
