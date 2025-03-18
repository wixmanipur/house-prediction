// Global array to store multiple images (map or otherwise)
let imagesList = [];

// Initialize a Chart.js bar chart for detection results
const ctx = document.getElementById('myChart')?.getContext('2d');
let myChart = null;

if (ctx) {
   myChart = new Chart(ctx, {
      type: 'bar',
      data: {
         labels: [],
         datasets: [{
            label: 'Detected Houses',
            data: [],
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
         }]
      },
      options: {
         responsive: true,
         scales: {
            y: {
               beginAtZero: true,
               title: {
                  display: true,
                  text: 'Number of Houses'
               }
            }
         }
      }
   });
}

// Utility to display original + annotated images side by side,
// with a title showing detection count.
function displayDetectionResult(name, original, annotated, count) {
   const container = document.createElement("div");
   container.classList.add("detection-result");

   // Title with image name + count
   const title = document.createElement("h3");
   title.textContent = `${name}: Detected ${count} houses`;
   container.appendChild(title);

   // Side-by-side images
   const imagesWrapper = document.createElement("div");
   imagesWrapper.classList.add("images-wrapper");

   const origImg = document.createElement("img");
   origImg.src = original;
   origImg.alt = "Original";
   imagesWrapper.appendChild(origImg);

   const annImg = document.createElement("img");
   annImg.src = annotated;
   annImg.alt = "Annotated";
   imagesWrapper.appendChild(annImg);

   container.appendChild(imagesWrapper);
   document.getElementById("resultsContainer").appendChild(container);
}

// 1) Initialize Google Maps (optional dynamic map)
async function initMap() {
   const lat = 24.792901;
   const lng = 93.933870;

   const radius = 1;
   const zoom = 8;

   const deltaLat = radius / 111;
   const deltaLng = radius / (111 * Math.cos(lat * Math.PI / 180));
   const swLat = lat - deltaLat;
   const swLng = lng - deltaLng;
   const neLat = lat + deltaLat;
   const neLng = lng + deltaLng;
   const visibleParam = `visible=${swLat},${swLng}|${neLat},${neLng}`;

   const apiKey = "AIzaSyCfB6DQNL8WrYegbR0Xga_T2JfXEovWGME"; // Replace with your actual API key
   const zoomParam = zoom ? `&zoom=${zoom}` : "";

   const url = `https://maps.googleapis.com/maps/api/staticmap?center=${lat},${lng}${zoomParam}&${visibleParam}&size=600x400&scale=2&maptype=satellite&key=${apiKey}`;

   document.getElementById("mapContainer").innerHTML =
      `<img src="${url}" alt="Satellite Map" />`;
}

// 2) Build a static map URL from single lat,lon input
function getMap() {
   const latLonInput = document.getElementById("latLon").value;
   const [latStr, lonStr] = latLonInput.split(",").map(item => item.trim());
   const lat = parseFloat(latStr);
   const lng = parseFloat(lonStr);

   const radius = parseFloat(document.getElementById("radius").value);
   const zoom = document.getElementById("zoom").value;
   const scale = document.getElementById("scale").value;

   if (isNaN(lat) || isNaN(lng) || isNaN(radius)) {
      alert("Please enter valid lat,lon (e.g. 24.642248, 94.098629) and radius.");
      return;
   }

   const deltaLat = radius / 111;
   const deltaLng = radius / (111 * Math.cos(lat * Math.PI / 180));
   const swLat = lat - deltaLat;
   const swLng = lng - deltaLng;
   const neLat = lat + deltaLat;
   const neLng = lng + deltaLng;
   const visibleParam = `visible=${swLat},${swLng}|${neLat},${neLng}`;

   const apiKey = "AIzaSyCfB6DQNL8WrYegbR0Xga_T2JfXEovWGME"; // Replace with your actual API key
   const zoomParam = zoom ? `&zoom=${zoom}` : "";
   const scaleParam = scale ? `&scale=${scale}` : "";

   const url = `https://maps.googleapis.com/maps/api/staticmap?center=${lat},${lng}${zoomParam}${scaleParam}&${visibleParam}&size=600x400&maptype=satellite&key=${apiKey}`;

   document.getElementById("mapContainer").innerHTML =
      `<img src="${url}" alt="Satellite Map" />`;
}

// 3) Add the current map image to imagesList + UI list
function addMapImage() {
   const mapImg = document.querySelector("#mapContainer img");
   if (!mapImg || !mapImg.src) {
      alert("Map image not found. Click 'Get Map' first.");
      return;
   }
   const name = "Map Image " + (imagesList.length + 1);
   imagesList.push({ name, src: mapImg.src });

   const li = document.createElement("li");
   li.textContent = name;
   document.getElementById("imageListUI").appendChild(li);

   alert(`Added "${name}" to the image list.`);
}

// 4) Upload a single file for detection
function uploadImage() {
   const input = document.getElementById('imageUpload');
   if (input.files.length === 0) {
      alert('Please select an image file.');
      return;
   }

   const file = input.files[0];
   document.getElementById('result').innerText = 'Detecting houses, please wait...';

   const reader = new FileReader();
   reader.onload = function (e) {
      detectHouseFromFile(file);
   };
   reader.readAsDataURL(file);
}

// 5) Send a file (or Blob) to the backend for detection using a relative URL
function detectHouseFromFile(fileObj, nameOverride = null) {
   const formData = new FormData();
   formData.append('file', fileObj);

   const conf = document.getElementById('confInput').value;
   const iou = document.getElementById('iouInput').value;
   const imgsz = document.getElementById('imgszInput').value;
   const draw_label = document.getElementById('drawLabelInput').checked;

   formData.append('conf', conf);
   formData.append('iou', iou);
   formData.append('imgsz', imgsz);
   formData.append('draw_label', draw_label);

   const fileName = nameOverride || fileObj.name || "Uploaded Image";

   fetch('/predict', {
      method: 'POST',
      body: formData
   })
      .then(response => response.json())
      .then(data => {
         if (data.error) {
            document.getElementById('result').innerText = data.error;
         } else {
            const count = data.predictions;
            document.getElementById('result').innerText = `Detected ${count} houses in ${fileName}`;
            displayDetectionResult(fileName, data.original_image, data.annotated_image, count);
            if (myChart) {
               myChart.data.labels.push(fileName);
               myChart.data.datasets[0].data.push(count);
               myChart.update();
            }
         }
      })
      .catch(error => {
         console.error('Error:', error);
         document.getElementById('result').innerText = 'An error occurred. Please try again.';
      });
}

// 6) Fetch an image URL as a Blob
function fetchImageFromURL(imageUrl) {
   return fetch(imageUrl)
      .then(response => response.blob())
      .catch(error => {
         console.error("Error fetching image:", error);
         return null;
      });
}

// 7) Detect houses from an image URL
async function detectHouseFromURL(imageUrl, name) {
   const blob = await fetchImageFromURL(imageUrl);
   if (!blob) return 0;

   return new Promise((resolve) => {
      detectHouseFromFile(blob, name);
      setTimeout(() => {
         resolve(0);
      }, 1000);
   });
}

// 8) Run detection on all images in imagesList
async function runDetectionOnAllImages() {
   if (imagesList.length === 0) {
      alert("No images in the list. Add images first.");
      return;
   }

   document.getElementById('result').innerText = 'Detecting houses for all images, please wait...';

   if (myChart) {
      myChart.data.labels = [];
      myChart.data.datasets[0].data = [];
      myChart.update();
   }

   document.getElementById("resultsContainer").innerHTML = "";

   for (let i = 0; i < imagesList.length; i++) {
      const { name, src } = imagesList[i];
      await detectHouseFromURL(src, name);
   }

   document.getElementById('result').innerText = 'Detection completed for all images.';
}

// 9) Upload multiple files for detection using a relative URL
function uploadMultipleImages() {
   const input = document.getElementById('imageUploadMultiple');
   if (input.files.length === 0) {
      alert('Please select at least one image file.');
      return;
   }

   document.getElementById('result').innerText = 'Detecting houses, please wait...';

   const formData = new FormData();
   for (let i = 0; i < input.files.length; i++) {
      formData.append('files', input.files[i]);
   }

   const conf = document.getElementById('confInput').value;
   const iou = document.getElementById('iouInput').value;
   const imgsz = document.getElementById('imgszInput').value;
   const draw_label = document.getElementById('drawLabelInput').checked;

   formData.append('conf', conf);
   formData.append('iou', iou);
   formData.append('imgsz', imgsz);
   formData.append('draw_label', draw_label);

   fetch('/predict_multiple', {
      method: 'POST',
      body: formData
   })
      .then(response => response.json())
      .then(data => {
         if (data.error) {
            document.getElementById('result').innerText = data.error;
         } else {
            document.getElementById('result').innerText = 'Detection completed for selected images.';
            data.results.forEach(item => {
               if (item.error) {
                  alert(`Error for ${item.filename}: ${item.error}`);
               } else {
                  displayDetectionResult(item.filename, item.original_image, item.annotated_image, item.predictions);
                  if (myChart) {
                     myChart.data.labels.push(item.filename);
                     myChart.data.datasets[0].data.push(item.predictions);
                     myChart.update();
                  }
               }
            });
         }
      })
      .catch(error => {
         console.error('Error:', error);
         document.getElementById('result').innerText = 'An error occurred. Please try again.';
      });
}
