const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', draw);

function startDrawing(e) {
    drawing = true;
    draw(e);  // In case the user just clicks without moving
}

function stopDrawing() {
    drawing = false;
    ctx.beginPath();  // Reset the drawing path
    sendCanvasDataForPrediction();  // After drawing stops, send the image
}

function draw(e) {
    if (!drawing) return;

    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predictions').innerHTML = '';  // Clear the predictions
}

async function sendCanvasDataForPrediction() {
    const imageData = canvas.toDataURL('image/png');

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
    });

    const data = await response.json();
    const probabilities = data.predictions;

    // Find the highest and second-highest probabilities
    let highestProb = -1;
    let secondHighestProb = -1;
    let highestDigit = -1;
    let secondHighestDigit = -1;

    for (let i = 0; i < probabilities.length; i++) {
        const prob = probabilities[i];

        if (prob > highestProb) {
            secondHighestProb = highestProb;
            secondHighestDigit = highestDigit;

            highestProb = prob;
            highestDigit = i;
        } else if (prob > secondHighestProb) {
            secondHighestProb = prob;
            secondHighestDigit = i;
        }
    }

    // Update predictions
    document.getElementById('most-probable').innerText = highestDigit;
    document.getElementById('most-probable-confidence').innerText = (highestProb * 100).toFixed(2);
    document.getElementById('second-probable').innerText = secondHighestDigit;
    document.getElementById('second-probable-confidence').innerText = (secondHighestProb * 100).toFixed(2);
}
