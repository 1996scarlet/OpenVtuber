////////////////////////////////////////////////////////////////////////////////////////////////
// ユーザーパラメータ

//アルファ値
float Alpha = 0.8;

//背景色
float4 ClearColor
<
   string UIName = "ClearColor";
   string UIWidget = "Color";
   bool UIVisible =  true;
> = float4(0,0,0,0);


texture EyeRT: OFFSCREENRENDERTARGET <
    string Description = "OffScreen RenderTarget for PostAlphakEye.fx";
    float4 ClearColor = { 0, 0, 0, 0 };
    float ClearDepth = 1.0;
    bool AntiAlias = true;
    int MipLevels = 1;
    string DefaultEffect = 
        
"物述有栖.pmx = aliceeye.fx;"

///////////////////////////////////////////////////////////////////////////////
        
        "* = DrawMask.fx;";
        
>;

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//これ以降はエフェクトの知識のある人以外は触れないこと



float Script : STANDARDSGLOBAL <
    string ScriptOutput = "color";
    string ScriptClass = "scene";
    string ScriptOrder = "postprocess";
> = 0.8;




sampler EyeView = sampler_state {
    texture = <EyeRT>;
    Filter = LINEAR;
    AddressU  = CLAMP;
    AddressV = CLAMP;
};


// マテリアル色
float4 MaterialDiffuse : DIFFUSE  < string Object = "Geometry"; >;
static float alpha1 = MaterialDiffuse.a;

float scaling0 : CONTROLOBJECT < string name = "(self)"; >;
static float scaling = scaling0 * 0.1;

float3 objpos : CONTROLOBJECT < string name = "(self)"; >;

// スクリーンサイズ
float2 ViewportSize : VIEWPORTPIXELSIZE;

static const float2 ViewportOffset = (float2(0.5,0.5)/ViewportSize);

// レンダリングターゲットのクリア値
float ClearDepth  = 1.0;


////////////////////////////////////////////////////////////////////////////////////////////////
// 共通頂点シェーダ
struct VS_OUTPUT {
    float4 Pos            : POSITION;
    float2 Tex            : TEXCOORD0;
};

VS_OUTPUT VS_passDraw( float4 Pos : POSITION, float4 Tex : TEXCOORD0 ) {
    VS_OUTPUT Out = (VS_OUTPUT)0; 
    
    Out.Pos = Pos;
    Out.Tex = Tex + ViewportOffset;
    
    return Out;
}

////////////////////////////////////////////////////////////////////////////////////////////////

float4 PS_Test( float2 Tex: TEXCOORD0 ) : COLOR {   
    float4 Color = tex2D(EyeView, Tex);
    Color.a *= (alpha1 * Alpha);
    return Color;
}

////////////////////////////////////////////////////////////////////////////////////////////////

technique PostAlphaEye <
    string Script = 
        
        "RenderColorTarget0=;"
        "RenderDepthStencilTarget=;"
        "ClearSetColor=ClearColor;"
        "ClearSetDepth=ClearDepth;"
        "Clear=Color;"
        "Clear=Depth;"
        "ScriptExternal=Color;"
        
        "RenderColorTarget0=;"
        "RenderDepthStencilTarget=;"
        "Pass=DrawEye;"
    ;
    
> {
    
    pass DrawEye < string Script= "Draw=Buffer;"; > {
        AlphaBlendEnable = true;
        VertexShader = compile vs_2_0 VS_passDraw();
        PixelShader  = compile ps_2_0 PS_Test();
    }
    
}
////////////////////////////////////////////////////////////////////////////////////////////////
