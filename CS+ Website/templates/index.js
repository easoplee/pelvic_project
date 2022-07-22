function showPelv() {
    /* Access image by id and change
    the display property to block*/
    if (document.getElementById('pel-image').style.display = "none") {
        document.getElementById('pel-image').style.display = "inline";

    } if (document.getElementById('pel-image').style.display = "inline") {
            document.getElementById('pel-image').style.display = "none"; 
        }
}

function show(id, visible) {
    var img = document.getElementById(id);
    img.style.display = (visible);
}