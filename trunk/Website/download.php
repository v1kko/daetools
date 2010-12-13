<?php
$filename = $_GET['filename'];

$counter = "counter.txt";

// Date logging;
$today = getdate();
$hour = $today[hours];
$min = $today[minutes];
$sec = $today[seconds];
$month = $today[mon];
$mday = $today[mday];
$year = $today[year];
$current_date = $mday . $month . $year;

// Log visit;
$fp = fopen($counter, "a");
$line = "[" . $filename . "] [" . getIP() . "] [" . $mday . "/" . $month . "/" . $year . ", " . $hour . ":" . $min . ":" . $sec . "] \n";

$size = strlen($line);
fputs($fp, $line, $size);
fclose($fp);

$downloadfile = "bin/".$filename;

if(!file_exists($downloadfile)) 
{
    die('The specified file does not exist');
} 


set_time_limit(0);  
output_file($downloadfile, $filename, 'application/octet-stream');

/*
$fsize = filesize($downloadfile);

header("Content-disposition: attachment; filename=$filename");
header("Content-type: application/octet-stream");
header("Content-length: $fsize");
//header("Cache-control: private"); //use this to open files directly
header('Pragma: no-cache');
header('Expires: 0');

readfile("$downloadfile");
*/


function getIP() {
  $IP = '';
  if (getenv('HTTP_CLIENT_IP')) {
    $IP =getenv('HTTP_CLIENT_IP');
  } elseif (getenv('HTTP_X_FORWARDED_FOR')) {
    $IP =getenv('HTTP_X_FORWARDED_FOR');
  } elseif (getenv('HTTP_X_FORWARDED')) {
    $IP =getenv('HTTP_X_FORWARDED');
  } elseif (getenv('HTTP_FORWARDED_FOR')) {
    $IP =getenv('HTTP_FORWARDED_FOR');
  } elseif (getenv('HTTP_FORWARDED')) {
    $IP = getenv('HTTP_FORWARDED');
  } else {
    $IP = $_SERVER['REMOTE_ADDR'];
  }
return $IP;
}

function output_file($file, $name, $mime_type='')
{
 /*
 This function takes a path to a file to output ($file), 
 the filename that the browser will see ($name) and 
 the MIME type of the file ($mime_type, optional).
 
 If you want to do something on download abort/finish,
 register_shutdown_function('function_name');
 */
 if(!is_readable($file)) die('File not found or inaccessible!');
 
 $size = filesize($file);
 $name = rawurldecode($name);
 
 /* Figure out the MIME type (if not specified) */
 $known_mime_types=array(
    "pdf" => "application/pdf",
    "txt" => "text/plain",
    "html" => "text/html",
    "htm" => "text/html",
    "exe" => "application/octet-stream",
    "zip" => "application/zip",
    "doc" => "application/msword",
    "xls" => "application/vnd.ms-excel",
    "ppt" => "application/vnd.ms-powerpoint",
    "gif" => "image/gif",
    "png" => "image/png",
    "jpeg"=> "image/jpg",
    "jpg" =>  "image/jpg",
    "php" => "text/plain"
 );
 
 if($mime_type==''){
     $file_extension = strtolower(substr(strrchr($file,"."),1));
     if(array_key_exists($file_extension, $known_mime_types)){
        $mime_type=$known_mime_types[$file_extension];
     } else {
        $mime_type="application/force-download";
     };
 };
 
 @ob_end_clean(); //turn off output buffering to decrease cpu usage
 
 // required for IE, otherwise Content-Disposition may be ignored
 if(ini_get('zlib.output_compression'))
  ini_set('zlib.output_compression', 'Off');
 
 header('Content-Type: ' . $mime_type);
 header('Content-Disposition: attachment; filename="'.$name.'"');
 header("Content-Transfer-Encoding: binary");
 header('Accept-Ranges: bytes');
 
 /* The three lines below basically make the 
    download non-cacheable */
 header("Cache-control: private");
 header('Pragma: private');
 header("Expires: Mon, 26 Jul 1997 05:00:00 GMT");
 
 // multipart-download and download resuming support
 if(isset($_SERVER['HTTP_RANGE']))
 {
    list($a, $range) = explode("=",$_SERVER['HTTP_RANGE'],2);
    list($range) = explode(",",$range,2);
    list($range, $range_end) = explode("-", $range);
    $range=intval($range);
    if(!$range_end) {
        $range_end=$size-1;
    } else {
        $range_end=intval($range_end);
    }
 
    $new_length = $range_end-$range+1;
    header("HTTP/1.1 206 Partial Content");
    header("Content-Length: $new_length");
    header("Content-Range: bytes $range-$range_end/$size");
 } else {
    $new_length=$size;
    header("Content-Length: ".$size);
 }
 
 /* output the file itself */
 $chunksize = 1*(1024*1024); //you may want to change this
 $bytes_send = 0;
 if ($file = fopen($file, 'r'))
 {
    if(isset($_SERVER['HTTP_RANGE']))
    fseek($file, $range);
 
    while(!feof($file) && 
        (!connection_aborted()) && 
        ($bytes_send<$new_length)
          )
    {
        $buffer = fread($file, $chunksize);
        print($buffer); //echo($buffer); // is also possible
        flush();
        $bytes_send += strlen($buffer);
    }
 fclose($file);
 } else die('Error - can not open file.');
 
die();
}   


?> 