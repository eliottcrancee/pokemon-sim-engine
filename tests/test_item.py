import pytest

from pokemon.item import HealHP, ItemCategory, ItemError, ItemRegistry, Items
from pokemon.pokemon import Pokedex, Pokemon, PokemonStatus

# --- Fixtures ---


@pytest.fixture
def shinx():
    """Creates a healthy Level 10 Shinx."""
    return Pokemon(Pokedex.Shinx, level=10)


@pytest.fixture
def injured_shinx(shinx: Pokemon):
    """Creates a Shinx with 1 HP."""
    shinx.hp = 1
    return shinx


@pytest.fixture
def fainted_shinx(shinx: Pokemon):
    """Creates a fainted Shinx."""
    shinx.hp = 0
    shinx.is_alive = False
    shinx.status = PokemonStatus.FAINTED
    return shinx


@pytest.fixture
def burned_shinx(shinx: Pokemon):
    """Creates a burned Shinx with full HP."""
    shinx.status = PokemonStatus.BURN
    return shinx


# --- Test Functions ---


def test_potion_heals_hp(injured_shinx: Pokemon):
    """Test that a Potion heals exactly 20 HP."""
    initial_hp = injured_shinx.hp  # Should be 10

    # Apply Potion
    messages = Items.Potion.use(injured_shinx)

    assert injured_shinx.hp == initial_hp + 20
    assert "recovered 20 HP" in messages[0].text


def test_super_potion_caps_at_max_hp(injured_shinx: Pokemon):
    """Test that healing does not exceed max_hp."""
    # Ensure healing 50 would exceed max HP if max is low,
    # or simply heal 50 if max is high.
    # Let's assume lvl 10 Shinx has > 10 HP but < 200 HP.

    injured_shinx.hp = injured_shinx.max_hp - 10  # 10 HP missing

    # Apply Super Potion (Heals 50)
    messages = Items.SuperPotion.use(injured_shinx)

    assert injured_shinx.hp == injured_shinx.max_hp
    assert "recovered 10 HP" in messages[0].text  # Should report actual amount


def test_cannot_heal_full_hp_pokemon(shinx):
    """Test that medicine cannot be used on a healthy pokemon."""
    assert Items.Potion.can_use(shinx) is False

    messages = Items.Potion.use(shinx)
    assert messages[0].text == "It won't have any effect."


def test_full_heal_cures_status(burned_shinx):
    """Test that Full Heal cures a status condition."""
    assert Items.FullHeal.can_use(burned_shinx) is True

    messages = Items.FullHeal.use(burned_shinx)

    assert burned_shinx.status == PokemonStatus.HEALTHY
    assert "cured of its status" in messages[0].text


def test_full_heal_useless_on_healthy(shinx):
    """Test that Full Heal cannot be used if no status exists."""
    assert Items.FullHeal.can_use(shinx) is False


def test_revive_works_on_fainted(fainted_shinx):
    """Test that Revive brings a pokemon back to life with 50% HP."""
    assert Items.Revive.can_use(fainted_shinx) is True

    messages = Items.Revive.use(fainted_shinx)

    assert fainted_shinx.is_alive is True
    assert fainted_shinx.status == PokemonStatus.HEALTHY
    assert fainted_shinx.hp == fainted_shinx.max_hp // 2
    assert "was revived" in messages[0].text


def test_max_revive_fully_restores(fainted_shinx):
    """Test that Max Revive brings a pokemon back with 100% HP."""
    Items.MaxRevive.use(fainted_shinx)

    assert fainted_shinx.is_alive is True
    assert fainted_shinx.hp == fainted_shinx.max_hp


def test_revive_cannot_use_on_living(injured_shinx):
    """Test that Revive cannot be used on a living pokemon."""
    assert Items.Revive.can_use(injured_shinx) is False


def test_item_registry_lookup():
    """Test that items can be retrieved via the Registry."""
    potion = ItemRegistry.get("Potion")
    assert potion.name == "Potion"

    # Test case insensitivity and spacing
    super_potion = ItemRegistry.get("super potion")
    assert super_potion.name == "Super Potion"


def test_registry_duplicate_error():
    """Test that registering an item with an existing name raises an error."""
    # We must try to register a name that already exists (e.g., 'Potion')
    with pytest.raises(ItemError):
        ItemRegistry.register("Potion", ItemCategory.MEDICINE, HealHP(10), "desc")


def test_items_accessor_attribute_error():
    """Test accessing a non-existent item via Items class."""
    with pytest.raises(ItemError):
        _ = Items.MasterSword
