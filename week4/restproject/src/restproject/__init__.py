from restproject.agent import (
    DEFAULT_MENU,
    RestaurantOrderState,
    build_restaurant_graph,
    create_initial_state,
    run_interactive,
    simulate_order,
)


def main() -> None:
    run_interactive()


__all__ = [
    "DEFAULT_MENU",
    "RestaurantOrderState",
    "build_restaurant_graph",
    "create_initial_state",
    "main",
    "run_interactive",
    "simulate_order",
]
