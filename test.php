<?php
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Command to execute the Python script
$command = "python Basic.py";

// Capture the output and return value
$output = [];
$return_var = 0;
exec($command, $output, $return_var);

// Display the command that was run
echo "<pre>";
echo "Command: $command\n";
echo "Output:\n";
print_r($output); // Display the output of the Python script
echo "\nReturn Value: $return_var\n";
echo "</pre>";
?>
