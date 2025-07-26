const letterImages = [
    { filename: "/static/images/size_6_60(1).png", sizeLabel: "6/60" },
    { filename: "/static/images/size_6_36(1).png", sizeLabel: "6/36" },
    { filename: "/static/images/size_6_24(1).png", sizeLabel: "6/24" },
    { filename: "/static/images/size_6_18(1).png", sizeLabel: "6/18" },
    { filename: "/static/images/size_6_12(1).png", sizeLabel: "6/12" },
    { filename: "/static/images/size_6_9(1).png", sizeLabel: "6/9" },
    { filename: "/static/images/size_6_6(1).png", sizeLabel: "6/6" },
];

let currentIndex = 0;
let lastClear = null;

function nextImage(isClear) {
    if (isClear) {
        lastClear = letterImages[currentIndex].sizeLabel;
    }

    currentIndex++;

    if (currentIndex < letterImages.length) {
        const currentImage = letterImages[currentIndex];
        document.getElementById("letterImage").src = currentImage.filename;
        document.getElementById("currentSizeLabel").textContent = currentImage.sizeLabel;
    } else {
        showResults();
    }
}

function showResults() {
    const result = document.getElementById("result");
    const conditionElement = document.getElementById("condition");
    const powerElement = document.getElementById("estimatedPower");

    let condition, estimatedPower;

    // Check the last image that was clearly visible
    if (!lastClear) {
        condition = "Severe myopia (short sightedness)";
        estimatedPower = "6.00 or worse";  // No need for the minus sign in the severe myopia condition
    } else if (lastClear === "6/6") {
        condition = "Normal vision or possible slight hyperopia (long sightedness)";
        estimatedPower = "You're vision is prefect!!! ";  // Fixed power for "6/6"
    } else {
        const powerRange = getPowerRange(lastClear);

        // If the user couldn't read the image, display the full power range (min to max) without the minus sign
        estimatedPower = `Range: ${Math.abs(powerRange.max).toFixed(2)}  to ${Math.abs(powerRange.min).toFixed(2)} `;
    }

    powerElement.textContent = `Estimated Power: ${estimatedPower}`;
    result.style.display = "block";
}



// Function to get power range based on the size label
function getPowerRange(sizeLabel) {
    const acuityToPower = {
        "6/60": { min: -6, max: -5 },
        "6/36": { min: -4, max: -3 },
        "6/24": { min: -3, max: -2 },
        "6/18": { min: -2, max: -1.5 },
        "6/12": { min: -1.5, max: -1 },
        "6/9": { min: -1, max: -0.5 },
        "6/6": { min: 0, max: 0.5 }
    };

    return acuityToPower[sizeLabel];
}

// Function to restart the test
function restartTest() {
    currentIndex = 0;
    lastClear = null;
    document.getElementById("result").style.display = "none";
    document.getElementById("letterImage").src = letterImages[currentIndex].filename;
    document.getElementById("currentSizeLabel").textContent = letterImages[currentIndex].sizeLabel;
}

// Function to handle the user's response
function handleUserResponse(isClear) {
    if (!isClear) {
        // Stop and show the results immediately if user can't read the current image
        lastClear = letterImages[currentIndex].sizeLabel;
        showResults();
    } else {
        // Proceed to the next image if the user can read the current one
        nextImage(isClear);
    }
}
