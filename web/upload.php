<?php
$target_dir = "./images/";
$target_file = dechex(rand(1,100000000));
$uploadOk = 0;
// Check if image file is a actual image or fake image
if(isset($_FILES["file"]["tmp_name"])) {
    $check = getimagesize($_FILES["file"]["tmp_name"]);
    if($check !== false) {
        $uploadOk = 1;
    } else {
        echo "File is not an image.";
        $uploadOk = 0;
    }
}
// Check if file already exists
if (file_exists($target_dir . $target_file)) {
    echo "Sorry, file already exists.";
    $uploadOk = 0;
}
// Check file size
if ($_FILES["file"]["size"] > 500000000) {
    echo "Sorry, your file is too large.";
    $uploadOk = 0;
}
// Check if $uploadOk is set to 0 by an error
if ($uploadOk == 0) {
    http_response_code(400);
    echo "Sorry, your file was not uploaded.";
// if everything is ok, try to upload file
} else {
    if (move_uploaded_file($_FILES["file"]["tmp_name"], $target_dir . $target_file)) {
        echo $target_file;
    } else {
        echo "Sorry, there was an error uploading your file.".$_FILES["file"]["tmp_name"];
    }
}
?>