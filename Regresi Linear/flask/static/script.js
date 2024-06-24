document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const inputField = document.getElementById('input');
    const inputValues = inputField.value.split(',').map(Number);
    
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ input: inputValues })
    })
    .then(response => response.json())
    .then(data => {
        const predictionsList = document.getElementById('predictions-list');
        predictionsList.innerHTML = '';
        if (data.predictions) {
            data.predictions.forEach(prediction => {
                const li = document.createElement('li');
                li.textContent = prediction;
                predictionsList.appendChild(li);
            });
        } else if (data.error) {
            const li = document.createElement('li');
            li.textContent = `Error: ${data.error}`;
            predictionsList.appendChild(li);
        }
    })
    .catch(error => console.error('Error:', error));
});
