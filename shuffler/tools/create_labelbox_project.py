import labelbox as lb
import argparse
import logging


def get_parser():
    parser = argparse.ArgumentParser(
        description='Exports stamps and pages to Labelbox NDJSON format.')
    parser.add_argument('--api_key',
                        required=True,
                        help='API_KEY for LabelBox.')
    parser.add_argument('--in_dataset_name',
                        required=True,
                        help='The dataset name on LabelBox.')
    parser.add_argument('--in_ontology_name',
                        required=True,
                        choices=['stamps_and_pages', 'stamps', 'pages'],
                        help='Ontology name on LabelBox.')
    parser.add_argument('--in_instructions_file_path',
                        help='Instructions file. Can be .txt or .pdf.')
    parser.add_argument('--out_project_name',
                        required=True,
                        help='Name of the created project.')
    parser.add_argument('--out_batch_prefix',
                        required=True,
                        help='Prefix for created batch names.')
    parser.add_argument(
        '--logging',
        default=20,
        type=int,
        choices={10, 20, 30, 40},
        help='Log debug (10), info (20), warning (30), error (40).')
    return parser


def create_labelbox_project(args):
    client = lb.Client(args.api_key)

    # Get or create the project.
    project = client.get_projects(
        where=lb.Project.name == args.out_project_name).get_one()
    if project is not None:
        logging.info('Found project %s', args.out_project_name)
    else:
        logging.info('Project %s does not exist, creating it.',
                     args.out_project_name)
        project = client.create_project(name=args.out_project_name,
                                        media_type=lb.MediaType.Image)

    if args.in_instructions_file_path is not None:
        project.upsert_instructions(args.in_instructions_file_path)
    print('Project:', project)

    # # Attach dataset to the project.
    # dataset = client.get_datasets(
    #     where=lb.Dataset.name == args.in_dataset_name).get_one()
    # task = project.create_batches_from_dataset(
    #     name_prefix=args.out_batch_prefix, dataset_id=dataset.uid, priority=1)
    # print("Errors: ", task.errors())
    # print("Result: ", task.result())

    # ontology = client.get_ontology('clyggpj3o05ij073l6l1i6bi0')
    ontology = next(client.get_ontologies(args.in_ontology_name))
    print(ontology)
    # FIXME: Does not work, don't know why. Had to do it manually in webapp.
    # https://community.labelbox.com/t/project-connect-ontology-ontology-throws-stopiteration/2801
    project.connect_ontology(ontology)


if __name__ == '__main__':
    args = get_parser().parse_args()
    logging.basicConfig(level=args.logging,
                        format='%(levelname)s: %(message)s')
    create_labelbox_project(args)
