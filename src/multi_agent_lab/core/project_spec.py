"""Structured functional specification for vague user goals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ProjectSpec:
    """Functional MVP specification generated before planning."""

    app_name: str
    app_type: str
    domain_summary: str
    entities: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    screens: list[str] = field(default_factory=list)
    backend_modules: list[str] = field(default_factory=list)
    frontend_modules: list[str] = field(default_factory=list)
    validations: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the spec to primitive values."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the spec as stable, readable JSON."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectSpec:
        """Deserialize a project spec from primitive values."""
        return cls(
            app_name=str(data.get("app_name", "MVP App")),
            app_type=str(data.get("app_type", "web_app")),
            domain_summary=str(data.get("domain_summary", "")),
            entities=[str(item) for item in data.get("entities", [])],
            features=[str(item) for item in data.get("features", [])],
            screens=[str(item) for item in data.get("screens", [])],
            backend_modules=[str(item) for item in data.get("backend_modules", [])],
            frontend_modules=[str(item) for item in data.get("frontend_modules", [])],
            validations=[str(item) for item in data.get("validations", [])],
            assumptions=[str(item) for item in data.get("assumptions", [])],
            out_of_scope=[str(item) for item in data.get("out_of_scope", [])],
        )


def sales_3d_printing_spec() -> ProjectSpec:
    """Return the deterministic MVP spec for 3D printing sales."""
    return ProjectSpec(
        app_name="3D Print Sales Tracker",
        app_type="web_app",
        domain_summary=(
            "MVP para registrar ventas de impresion 3D, clientes, productos, "
            "lineas de venta y estado de pago."
        ),
        entities=["Product", "Customer", "Sale", "SaleItem", "Payment"],
        features=[
            "CRUD productos",
            "CRUD clientes",
            "registrar venta",
            "listar ventas",
            "resumen mensual",
            "estado de pago",
        ],
        screens=["Dashboard", "Products", "Customers", "Sales", "New Sale"],
        backend_modules=[
            "products",
            "customers",
            "sales",
            "payments",
            "monthly_summary",
        ],
        frontend_modules=[
            "dashboard",
            "product_management",
            "customer_management",
            "sales_list",
            "new_sale_form",
        ],
        validations=[
            "total >= 0",
            "quantity > 0",
            "sale date required",
            "customer optional",
            "payment status required",
        ],
        assumptions=[
            "MVP local para una sola persona usuaria.",
            "Los importes se registran manualmente.",
            "El cliente puede omitirse para ventas anonimas.",
        ],
        out_of_scope=[
            "Facturacion fiscal avanzada.",
            "Inventario automatico de filamento.",
            "Pasarelas de pago reales.",
            "Multiusuario y permisos avanzados.",
        ],
    )


def generic_mvp_spec(goal: str) -> ProjectSpec:
    """Return a small generic MVP spec for unsupported vague goals."""
    return ProjectSpec(
        app_name="MVP App",
        app_type="web_app",
        domain_summary=f"MVP razonable para: {goal}",
        entities=["User", "Item"],
        features=["crear registros", "listar registros", "editar registros"],
        screens=["Dashboard", "Items"],
        backend_modules=["items"],
        frontend_modules=["dashboard", "items"],
        validations=["name required"],
        assumptions=["Se prioriza una version simple y local."],
        out_of_scope=["SaaS completo", "Integraciones externas", "Autenticacion avanzada"],
    )
