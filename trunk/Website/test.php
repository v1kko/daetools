<?php
foreach ( array('tar','gtar','gzip','gunzip') as $bin) {
    exec("whereis $bin", $ret);
    print_r($ret);
}
?>
