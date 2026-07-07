#include <exception>
#include <iostream>

// OpenCASCADE Foundation & Mathematics
#include <Standard_Version.hxx>
#include <gp_Ax2.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>

// OpenCASCADE Topological Shape Definition
#include <TopoDS_Shape.hxx>

// OpenCASCADE Geometric Primitives
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>

// OpenCASCADE Boolean Operations
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepBuilderAPI_Transform.hxx>

// OpenCASCADE XDE/CAF & Colors/Materials
#include <BinXCAFDrivers.hxx>
#include <Quantity_Color.hxx>
#include <Quantity_ColorRGBA.hxx>
#include <TCollection_ExtendedString.hxx>
#include <TCollection_HAsciiString.hxx>
#include <TDocStd_Application.hxx>
#include <TDocStd_Document.hxx>
#include <XCAFDoc_ColorTool.hxx>
#include <XCAFDoc_ColorType.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>

// OpenCASCADE Visual Materials (PBR)
#include <XCAFDoc_VisMaterial.hxx>
#include <XCAFDoc_VisMaterialPBR.hxx>
#include <XCAFDoc_VisMaterialTool.hxx>

// OpenCASCADE glTF Exporter
#include <Message_ProgressRange.hxx>
#include <NCollection_IndexedDataMap.hxx>
#include <RWGltf_CafWriter.hxx>
#include <TCollection_AsciiString.hxx>

// OpenCASCADE Mesh & Export
#include <BRepMesh_IncrementalMesh.hxx>
#include <Interface_Static.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeCone.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <gp_Pnt.hxx>
#include <gp_Dir.hxx>
#include <gp_Ax2.hxx>


int main() {
  std::cout << "Using OpenCASCADE Version: " << OCC_VERSION_STRING << std::endl;
  std::cout << "Creating a mechanical block with PBR material (scaled by 1000 "
               "to meters)..."
            << std::endl;

  try {
    // 1. Create the Shaft (Cylinder)
    // Bottom center at (0, 0, 0), radius = 5mm, height = 80mm
    gp_Pnt shaftOrigin(0.0, 0.0, 0.0);
    gp_Dir shaftDir(0.0, 0.0, 1.0); // Pointing up along Z
    gp_Ax2 shaftAxis(shaftOrigin, shaftDir);
    TopoDS_Shape shaft = BRepPrimAPI_MakeCylinder(shaftAxis, 5.0, 80.0).Shape();
    // 2. Create the Arrowhead Tip (Cone)
    // Start the base of the cone exactly where the shaft ends: Z = 80mm
    // Bottom radius = 15mm (flared out), top radius = 0mm (sharp tip), height =
    // 30mm
    gp_Pnt tipOrigin(0.0, 0.0, 80.0);
    gp_Ax2 tipAxis(tipOrigin, shaftDir);
    TopoDS_Shape tip = BRepPrimAPI_MakeCone(tipAxis, 15.0, 0.0, 30.0).Shape();
    // 3. Weld them together (Boolean Fuse)
    BRepAlgoAPI_Fuse fuser(shaft, tip);
    if (!fuser.IsDone()) {
      std::cerr << "Error: Failed to merge arrow shaft and tip!" << std::endl;
      return 1;
    }
    TopoDS_Shape unifiedArrow = fuser.Shape(); // This is now a SINGLE solid

    // 1. Create a Box: width = 100mm, depth = 50mm, height = 30mm
    BRepPrimAPI_MakeBox boxMaker(100.0, 50.0, 30.0);
    TopoDS_Shape box = boxMaker.Shape();

    // 2. Create a Cylinder to cut out of the box (Diameter = 25mm)
    gp_Pnt cylinderOrigin(10.0, 25.0, -5.0);
    gp_Dir cylinderDirection(0.0, 0.0, 1.0);
    gp_Ax2 cylinderAxis(cylinderOrigin, cylinderDirection);

    double radius = 12.5;
    double height = 40.0;
    BRepPrimAPI_MakeCylinder cylinderMaker(cylinderAxis, radius, height);
    TopoDS_Shape cylinder = cylinderMaker.Shape();

    // 3. Perform a Boolean Cut (Box minus Cylinder)
    std::cout << "Performing Boolean Cut (drilling the hole)..." << std::endl;
    BRepAlgoAPI_Cut cutOperation(box, cylinder);
    if (!cutOperation.IsDone()) {
      std::cerr << "Error: Boolean operation failed!" << std::endl;
      return 1;
    }
    TopoDS_Shape mechanicalPart = cutOperation.Shape();

    // 4. Scale shape by 1/1000 to convert mm to meters (required for standard
    // glTF scale)
    std::cout << "Scaling shape to meters for glTF export..." << std::endl;
    gp_Trsf scaleTransform;
    scaleTransform.SetScale(gp_Pnt(0.0, 0.0, 0.0), 0.001); // 1mm -> 0.001m
    BRepBuilderAPI_Transform transformOperation(unifiedArrow, scaleTransform);
    TopoDS_Shape scaledPart = transformOperation.Shape();

    // 5. Mesh the scaled shape (Tessellation is required for glTF)
    std::cout << "Generating triangular mesh on scaled shape..." << std::endl;
    // Since model is 1000x smaller, we reduce the linear deflection tolerance
    // by 1000 too (0.1mm -> 0.0001m)
    double linearDeflection = 0.0001;
    BRepMesh_IncrementalMesh meshGenerator(scaledPart, linearDeflection);
    meshGenerator.Perform();

    // 6. Initialize XCAF application and document
    Handle(TDocStd_Application) app = new TDocStd_Application();
    BinXCAFDrivers::DefineFormat(app);

    Handle(TDocStd_Document) doc;
    app->NewDocument(TCollection_ExtendedString("BinXCAF"), doc);

    // Get tools for manipulating shapes and materials
    Handle(XCAFDoc_ShapeTool) shapeTool =
        XCAFDoc_DocumentTool::ShapeTool(doc->Main());
    Handle(XCAFDoc_ColorTool) colorTool =
        XCAFDoc_DocumentTool::ColorTool(doc->Main());
    Handle(XCAFDoc_VisMaterialTool) visMaterialTool =
        XCAFDoc_DocumentTool::VisMaterialTool(doc->Main());

    // Add the scaled shape to the document structure
    TDF_Label shapeLabel = shapeTool->AddShape(scaledPart);

    // 7. Define and set the VISUAL PBR Material (Polished Aluminum)
    std::cout
        << "Assigning visual PBR material properties (Polished Aluminum)..."
        << std::endl;
    XCAFDoc_VisMaterialPBR pbrParams;
    pbrParams.BaseColor = Quantity_ColorRGBA(
        0.91f, 0.92f, 0.92f, 1.0f); // Bright white-silver albedo
    pbrParams.Metallic = 1.0f;      // Pure metal
    pbrParams.Roughness = 0.15f;    // Highly polished
    pbrParams.IsDefined = true;

    Handle(XCAFDoc_VisMaterial) visMaterial = new XCAFDoc_VisMaterial();
    visMaterial->SetPbrMaterial(pbrParams);

    TDF_Label visMatLabel =
        visMaterialTool->AddMaterial(visMaterial, "Polished_Aluminum");
    visMaterialTool->SetShapeMaterial(shapeLabel, visMatLabel);

    // Define a fallback basic color for older viewers
    colorTool->SetColor(shapeLabel,
                        Quantity_Color(0.91, 0.92, 0.92, Quantity_TOC_RGB),
                        XCAFDoc_ColorGen);

    // 8. Export to glTF / GLB (PBR mesh format)
    std::cout << "Exporting to glTF/GLB format..." << std::endl;
    NCollection_IndexedDataMap<TCollection_AsciiString, TCollection_AsciiString>
        metadata;
    metadata.Add(TCollection_AsciiString("Author"),
                 TCollection_AsciiString("OpenCASCADE User"));

    RWGltf_CafWriter gltfWriter("mechanical_part.glb",
                                true); // 'true' triggers binary .glb output

    if (gltfWriter.Perform(doc, metadata, Message_ProgressRange())) {
      std::cout << "-> Successfully saved 'mechanical_part.glb' (PBR settings, "
                   "scaled to meters)"
                << std::endl;
    } else {
      std::cerr << "-> Error exporting glTF/GLB file!" << std::endl;
      return 1;
    }

    std::cout << "All operations completed successfully!" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "An exception occurred: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
