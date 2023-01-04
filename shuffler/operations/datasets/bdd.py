import os.path as op
import cv2
import logging
from glob import glob
import simplejson as json
from progressbar import progressbar

from shuffler.backend import backend_media
from shuffler.utils import general as general_utils
from shuffler.utils import boxes as boxes_utils


def add_parsers(subparsers):
    importBddParser(subparsers)


def importBddParser(subparsers):
    parser = subparsers.add_parser(
        'importBdd',
        description='Import BDD annotations into a db. '
        'Both image-level and object-level attributes are written to the '
        '"properties" table. "manualShape" and "manualAttributes" are ignored. '
        'Objects with open polygons are ignored.')
    parser.set_defaults(func=importBdd)
    parser.add_argument('--images_dir',
                        required=True,
                        help='Directory with .jpg images.'
                        'E.g. "/my/path/to/BDD/bdd100k/seg/images/val". ')
    parser.add_argument(
        '--detection_json',
        help='Directory with .json annotations of objects. '
        'E.g. "/my/path/to/BDD/bdd100k/labels/bdd100k_labels_images_val.json"')
    parser.add_argument('--segmentation_dir',
                        help='Directory with .png segmentation masks.'
                        'E.g. "/my/path/to/BDD/bdd100k/seg/labels/val".')
    parser.add_argument('--display', action='store_true')


def importBdd(c, args):
    if args.display:
        imreader = backend_media.MediaReader(args.rootdir)

    image_paths = sorted(glob(op.join(args.images_dir, '*.jpg')))
    logging.info('Found %d JPG images in %s', len(image_paths),
                 args.images_dir)

    if args.detection_json:
        if not op.exists(args.detection_json):
            raise FileNotFoundError('Annotation file not found at "%s".' %
                                    args.detection_json)
        logging.info(
            'Loading the json with annotations. This may take a few seconds.')
        with open(args.detection_json) as f:
            detections = json.load(f)
            # Dict with image name as the key.
            detections = {d['name']: d for d in detections}

    for image_path in progressbar(image_paths):
        filename = op.splitext(op.basename(image_path))[0]
        logging.debug('Processing image: "%s"', filename)

        # Add image to the database.
        imheight, imwidth = backend_media.getPictureSize(image_path)
        imagefile = op.relpath(image_path, args.rootdir)
        c.execute('INSERT INTO images(imagefile,width,height) VALUES (?,?,?)',
                  (imagefile, imwidth, imheight))

        if args.display:
            img = imreader.imread(imagefile)

        # Detection annotations.
        if args.detection_json:
            imagename = op.basename(imagefile)
            if imagename not in detections:
                logging.error('Cant find image name "%s" in "%s"',
                              args.detection_json, imagename)
                continue

            detections_for_image = detections[imagename]
            image_properties = detections_for_image['attributes']
            for object_ in detections_for_image['labels']:

                object_bddid = object_['id']
                object_name = object_['category']
                object_properties = {
                    key: value
                    for key, value in object_['attributes'].items()
                    if value != 'none'
                }
                object_properties.update(image_properties)

                # Skip 3d object. TODO: import it to properties.
                if 'box3d' in object_:
                    logging.warning('Will skip 3D object %d.', object_bddid)
                    continue

                # Get the bbox if exists.
                x1 = y1 = width = height = None
                if 'box2d' in object_:
                    box2d = object_['box2d']
                    x1 = int(float(box2d['x1']))
                    y1 = int(float(box2d['y1']))
                    width = int(float(box2d['x2']) - x1)
                    height = int(float(box2d['y2']) - y1)
                    if args.display:
                        roi = boxes_utils.bbox2roi((x1, y1, width, height))
                        general_utils.drawScoredRoi(img, roi, object_name)

                c.execute(
                    'INSERT INTO objects(imagefile,x1,y1,width,height,name) '
                    'VALUES (?,?,?,?,?,?)',
                    (imagefile, x1, y1, width, height, object_name))
                objectid = c.lastrowid

                # Get the polygon if it exists.
                if 'poly2d' in object_:
                    if len(object_['poly2d']) > 1:
                        assert 0, len(object_['poly2d'])
                    polygon = object_['poly2d'][0]
                    polygon_name = None if polygon['closed'] else 'open_loop'
                    for pt in polygon['vertices']:
                        c.execute(
                            'INSERT INTO polygons(objectid,x,y,name) '
                            'VALUES (?,?,?,?)',
                            (objectid, pt[0], pt[1], polygon_name))
                    if args.display:
                        general_utils.drawScoredPolygon(
                            img, [(int(x[0]), int(x[1]))
                                  for x in polygon['vertices']], object_name)

                # Insert image-level and object-level attributes into
                # "properties" table.
                for key, value in object_properties.items():
                    c.execute(
                        'INSERT INTO properties(objectid,key,value) VALUES (?,?,?)',
                        (objectid, key, value))

        # Segmentation annotations.
        if args.segmentation_dir:
            segmentation_path = op.join(args.segmentation_dir,
                                        '%s_train_id.png' % filename)
            if not op.exists(segmentation_path):
                raise FileNotFoundError('Annotation file not found at "%s".' %
                                        segmentation_path)

            # Add image to the database.
            maskfile = op.relpath(segmentation_path, args.rootdir)
            c.execute('UPDATE images SET maskfile=? WHERE imagefile=?',
                      (maskfile, imagefile))

            if args.display:
                mask = imreader.maskread(maskfile)
                img = general_utils.drawMaskAside(img, mask, labelmap=None)

        # Maybe display.
        if args.display:
            cv2.imshow('importKitti', img[:, :, ::-1])
            if cv2.waitKey(-1) == 27:
                args.display = False
                cv2.destroyWindow('importKitti')
