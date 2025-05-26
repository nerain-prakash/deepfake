document.addEventListener("DOMContentLoaded", function() {
    const fileInput = document.querySelector('input[type="file"]');
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            alert(`Selected file: ${fileInput.files[0].name}`);
        }
    });
});
