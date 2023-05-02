const imageInput = document.getElementById("select-file-image");
const imgArea = document.querySelector(".img-area");
const output = document.getElementById("output");

// function to display the selected image
imageInput.addEventListener("change", function (event) {
  const file = event.target.files[0];
  const reader = new FileReader();

  reader.onload = function (event) {
    const img = new Image();
    img.src = event.target.result;
    img.onload = function () {
      imgArea.style.backgroundImage = `url(${img.src})`;
      imgArea.dataset.img = img.src;
    };
  };

  reader.readAsDataURL(file);
});

const outputContainer = document.querySelector(".outputContainer");
const uploadedImg = document.getElementById("uploaded-img");

function previewImage(event) {
  const file = event.target.files[0];
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    uploadedImg.src = reader.result;
    outputContainer.style.display = "block";
  };
}

// function to send image to server for classification
function classifyImage() {
  const imgData = imgArea.dataset.img;

  if (!imgData) {
    alert("Please select an image to classify.");
    return;
  }

  // send image data to server via AJAX
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/classify");
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xhr.onload = function () {
    if (xhr.status === 200) {
      const response = JSON.parse(xhr.responseText);
      showResult(response.result);
    } else {
      alert("Error: " + xhr.statusText);
    }
  };
  xhr.onerror = function () {
    alert("Error: " + xhr.statusText);
  };
  xhr.send(JSON.stringify({ image_data: imgData }));
}

// function to display classification results
function showResult(result) {
  output.innerHTML = `
    <h3>Classification Result:</h3>
    <ul>
      <li><strong>Malignant:</strong> ${result.malignant_probability.toFixed(
        2
      )}</li>
      <li><strong>Benign:</strong> ${result.benign_probability.toFixed(2)}</li>
    </ul>
  `;
}
