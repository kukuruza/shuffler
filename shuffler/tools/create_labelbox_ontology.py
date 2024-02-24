import labelbox as lb

import logging
import argparse
import logging
import json


def get_parser():
    parser = argparse.ArgumentParser(
        description='Creates an ontology on LabelBox.')
    parser.add_argument(
        '--api_key',
        required=True,
        help='API_KEY for LabelBox. If given will upload labels.')
    parser.add_argument(
        '--ontology_name',
        required=True,
        help='The ontology name on LabelBox. Should reflect classes.')
    parser.add_argument(
        '--use_stamps',
        action='store_true',
        help='Create the "stamp" class with the property "stamp_name".')
    parser.add_argument(
        '--use_pages',
        action='store_true',
        help='Create the "page" class with the property "page_type".')
    parser.add_argument('--out_ontology_file',
                        required=True,
                        help='The path to ontology json file on local disk.')
    parser.add_argument('--dry_run',
                        action='store_true',
                        help='Do not upload to Labelbox or write a file.')
    parser.add_argument(
        '--logging',
        default=20,
        type=int,
        choices={10, 20, 30, 40},
        help='Log debug (10), info (20), warning (30), error (40).')
    return parser


def create_labelbox_ontology(args):
    classifications = [
        lb.Classification(class_type=lb.Classification.Type.TEXT,
                          name="image_id")
    ]
    tools = []  # List of Tool objects
    if args.use_stamps:
        tools.append(
            lb.Tool(tool=lb.Tool.Type.BBOX,
                    name="stamp",
                    classifications=[
                        lb.Classification(
                            class_type=lb.Classification.Type.TEXT,
                            name="stamp_name"),
                    ]))
    if args.use_pages:
        tools.append(
            lb.Tool(tool=lb.Tool.Type.POLYGON,
                    name="page",
                    classifications=[
                        lb.Classification(
                            class_type=lb.Classification.Type.RADIO,
                            name="page_type",
                            options=[
                                lb.Option(value="page"),
                                lb.Option(value="pageb"),
                                lb.Option(value="pager"),
                                lb.Option(value="pagel"),
                                lb.Option(value="pagerb"),
                                lb.Option(value="pagelb"),
                            ]),
                    ]))
    ontology_builder = lb.OntologyBuilder(classifications=classifications,
                                          tools=tools)

    if not args.dry_run:
        client = lb.Client(args.api_key)
        ontology = client.create_ontology(args.ontology_name,
                                          ontology_builder.asdict(),
                                          media_type=lb.MediaType.Image)

        if args.out_ontology_file:
            logging.info('Writing ontology to "%s"', args.out_ontology_file)
            with open(args.out_ontology_file, 'w') as f:
                json.dump(ontology.normalized, f, indent=4)


if __name__ == '__main__':
    args = get_parser().parse_args()
    logging.basicConfig(level=args.logging,
                        format='%(levelname)s: %(message)s')

    create_labelbox_ontology(args)
