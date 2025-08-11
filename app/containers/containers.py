from dependency_injector import containers, providers
from app.core.config import Settings
from app.repositories.UserRepository import UserRepository
from app.services.user_service import UserService
from app.database.mongo import get_mongo_db
from app.database.postgres import get_postgres_db_session
from app.controllers.user_controller import UserController


class Container(containers.DeclarativeContainer):
    # wire modeles
    wiring_config = containers.WiringConfiguration(
        modules=[
            "app.routes.user_routes",
            "app.routes.prediction_router",
        ]
    )

    # configuration
    config = providers.Singleton(Settings)
    # database container
    mongo_db = providers.Singleton(get_mongo_db, config=config)
    postgres_db = providers.Singleton(get_postgres_db_session, config=config)

    # user
    user_repository = providers.Singleton(UserRepository,db=mongo_db)
    user_service = providers.Factory(UserService, repo=user_repository)
    user_controller = providers.Factory(UserController, service=user_service)
