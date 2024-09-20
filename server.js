const imageUpload = document.getElementById("imageUpload");
const loading = document.getElementById("loading"); // Reference to the spinner
let detectionText = null; // Variable to hold detection count text

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
])
  .then(() => {
    console.log("Models successfully loaded");
    start();
  })
  .catch((error) => console.error("Error loading models: ", error));

async function start() {
  const container = document.getElementById("container");
  let image, canvas;

  imageUpload.addEventListener("change", async () => {
    console.log("Image uploaded, starting processing...");

    // Clear previous image, canvas, and detection text
    if (image) image.remove();
    if (canvas) canvas.remove();
    if (detectionText) detectionText.remove(); // Remove previous face detection count

    try {
      // Show the loading spinner
      loading.style.display = "block";

      // Load the uploaded image
      image = await faceapi.bufferToImage(imageUpload.files[0]);
      console.log("Image loaded:", image);
      container.appendChild(image);

      // Create canvas from the image and overlay it
      canvas = faceapi.createCanvasFromMedia(image);
      container.appendChild(canvas);

      // Match canvas size with the image
      const displaySize = { width: image.width, height: image.height };
      faceapi.matchDimensions(canvas, displaySize);
      console.log("Canvas created and matched to image size:", displaySize);

      // Clear any previous results on the canvas
      const context = canvas.getContext("2d");
      context.clearRect(0, 0, canvas.width, canvas.height);

      // Detect faces and landmarks
      const detections = await faceapi
        .detectAllFaces(image)
        .withFaceLandmarks()
        .withFaceDescriptors();
      console.log("Face detections:", detections);

      // Create and append detection text to show number of detected faces
      detectionText = document.createElement("div");
      detectionText.innerText = `Detected Faces: ${detections.length}`;
      document.body.appendChild(detectionText);
      console.log(`Detected ${detections.length} faces`);

      // Resize detections and draw them on the canvas
      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      const faceMatcher = new faceapi.FaceMatcher(
        await loadLabeledImages(),
        0.6
      );
      const results = resizedDetections.map((d) =>
        faceMatcher.findBestMatch(d.descriptor)
      );

      // Draw face boxes and labels
      results.forEach((result, i) => {
        const box = resizedDetections[i].detection.box;
        console.log("Drawing box for face:", box);
        const drawBox = new faceapi.draw.DrawBox(box, {
          label: result.toString(),
        });
        drawBox.draw(canvas);
      });
    } catch (error) {
      console.error("Error processing the image:", error);
    } finally {
      // Hide the loading spinner when done
      loading.style.display = "none";
    }
  });
}

async function loadLabeledImages() {
  const labels = [
    "AB",
    "Aron Finch",
    "Chris Gayle",
    "Dale Steyn",
    "DK",
    "FAF DuPlesis",
    "Glen Maxwell",
    "Harshal Patel",
    "Josh Hazelwood",
    "Mohomad Siraj",
    "Shabaz Ahamed",
    "Virat Kohli",
    "Wanidu Hasaranga",
    "Yuzvendra Chahal",
  ];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        try {
          const img = await faceapi.fetchImage(
            `https://raw.githubusercontent.com/epcm18/FaceRecognition/main/samples_new/${label}/${i}.jpg`
          );
          console.log(`Fetching image for ${label}:`, img);
          const detection = await faceapi
            .detectSingleFace(img)
            .withFaceLandmarks()
            .withFaceDescriptor();
          if (detection) {
            descriptions.push(detection.descriptor);
            console.log(`Descriptor for ${label}:`, detection.descriptor);
          } else {
            console.warn(`No face detected in image ${i} for ${label}`);
          }
        } catch (error) {
          console.error(
            `Error fetching or detecting face for ${label}:`,
            error
          );
        }
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
