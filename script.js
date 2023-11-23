const form = document.getElementById('upload-form');
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    try {
        const response = await fetch('http://127.0.0.1:8000/predict/', {
            method: 'POST',
            body: formData
        });
        if (!response.ok) {
            throw new Error('Network response was not ok.');
        }
        const result = await response.json();

        // Open a new window to display the predicted class
        const newWindow = window.open("", "_blank", "width=400,height=200");
        newWindow.document.write('<h2 style="text-align: center;">Predicted Class:</h2>');
        newWindow.document.write('<p style="text-align: center;">' + result.predicted_class + '</p>');
    } catch (error) {
        console.error('Error:', error);
    }
});
