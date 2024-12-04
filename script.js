document.getElementById("uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();
  
    const fileInput = document.getElementById("imageInput");
    if (!fileInput.files[0]) {
      alert("Please select an image file.");
      return;
    }
  
    const formData = new FormData();
    formData.append("image", fileInput.files[0]);
  
    const resultDiv = document.getElementById("result");
    resultDiv.textContent = "Detecting...";
  
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
  
      if (!response.ok) {
        throw new Error("Failed to detect shape.");
      }
  
      const data = await response.json();
      resultDiv.textContent = `Detected Shape: ${data.label}`;
    } catch (error) {
      console.error("Error:", error);
      resultDiv.textContent = "An error occurred while detecting the shape.";
    }
  });