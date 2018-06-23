import os, sys
import os.path as op
import numpy as np
import cv2
import logging
import abc
import ConfigParser



def getCalibration ():
    '''
    Some scripts need a user to press keys.
    These keys may differ across platforms (win/linux/mac).
    This scripts performs calibration, storing, and loading user settings.
    '''
    # make a config file, if it doesn't exist
    config_path = op.join(os.getenv('HOME'), '.shuffler_config.ini')

    # try to read keys from the file
    config = ConfigParser.ConfigParser()
    if op.exists (config_path):
        config.read(config_path)
        try:
           keys_dict = {}
           keys_dict['del']   = int(config.get('opencv_keys', 'del'))
           keys_dict['right'] = int(config.get('opencv_keys', 'right'))
           keys_dict['left']  = int(config.get('opencv_keys', 'left'))
           keys_dict['esc']   = int(config.get('opencv_keys', 'esc'))
           return keys_dict
        except:
           logging.info ('will calibrate the keys')

    # if keys can't be read, then calibrate
    config = ConfigParser.ConfigParser()
    cv2.imshow('dummy', np.zeros((10,10), dtype=np.uint8))
    config.add_section('opencv_keys')

    print ('please click on the opencv window and click "del"')
    keydel = cv2.waitKey(-1)
    config.set('opencv_keys', 'del', keydel)
    
    print ('please click on the opencv window and click "right" or "+" or "pageup" as you prefer')
    keyright = cv2.waitKey(-1)
    config.set('opencv_keys', 'right', keyright)
    
    print ('please click on the opencv window and click "left" or "-" or "pagedown" as you prefer')
    keyleft = cv2.waitKey(-1)
    config.set('opencv_keys', 'left', keyleft)
    
    print ('please click on the opencv window and click "escape"')
    keyesc = cv2.waitKey(-1)
    config.set('opencv_keys', 'esc', keyesc)

    with open(config_path, 'w') as configfile:
        config.write(configfile)
    return { 'del': keydel, 'right': keyright, 'left': keyleft, 'esc': keyesc }


class KeyReaderUser:
    ''' Read keys from user '''
    def readKey(self):
        return cv2.waitKey(-1)


if __name__ == "__main__":
    getCalibration()
