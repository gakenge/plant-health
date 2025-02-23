// Load the Teachable Machine model
let model;
async function loadModel() {
  model = await tf.loadLayersModel('model/model.json');
  console.log('Model loaded successfully!');
}

// Handle image upload and preview
const imageUpload = document.getElementById('image-upload');
const imagePreview = document.getElementById('image-preview');
const predictButton = document.getElementById('predict-button');
const resultsDiv = document.getElementById('results');

imageUpload.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      imagePreview.style.display = 'block';
      predictButton.disabled = false;
    };
    reader.readAsDataURL(file);
  }
});

// Run prediction on the uploaded image
predictButton.addEventListener('click', async () => {
  if (!model) {
    alert('Model is still loading. Please wait.');
    return;
  }

  // Preprocess the image
  const image = tf.browser.fromPixels(imagePreview)
    .resizeNearestNeighbor([224, 224]) // Resize to the input size of the model
    .toFloat()
    .expandDims();

  // Normalize the image (if required by your model)
  const normalizedImage = image.div(255.0);

  // Run prediction
  const predictions = await model.predict(normalizedImage);
  const results = await predictions.data();

  // Display results
  const classNames = ['Healthy', 'Diseased']; // Replace with your class names
  let output = '<h2>Results:</h2>';
  classNames.forEach((className, index) => {
    output += `<p>${className}: ${(results[index] * 100).toFixed(2)}%</p>`;
  });
  resultsDiv.innerHTML = output;
});

// Load the model when the page loads
loadModel();